"""Bounded one-batch lookahead for a separate OPD rollout process.

Only CPU tensors travel on the dedicated Gloo group. A learner's I/O thread
receives the next cache while its main thread runs CUDA backward. CUDA work and
model snapshots remain on the main thread; no live model is shared with workers.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time

import torch
import torch.distributed as dist

HEADER, WEIGHTS, ACK = 310, 311, 312


def capture_snapshot(named):
    """One immutable CPU copy may be shared by multiple actor transport threads."""
    return [(kind, name, p.detach().to(device="cpu", copy=True).contiguous())
            for kind, name, p in named]


class ActorTimings:
    """Wall times plus CUDA stream spans; overlapping spans must not be summed."""
    def __init__(self, device):
        self.device = device
        self.wall, self.events = {}, []

    @contextmanager
    def measure(self, name):
        cuda = self.device.type == "cuda"
        if cuda:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.wall[name] = self.wall.get(name, 0.) + time.perf_counter() - start
            if cuda:
                b.record()
                self.events.append((name, a, b))

    def finish(self):
        result = {name + "_wall": value for name, value in self.wall.items()}
        if self.events:
            torch.cuda.synchronize(self.device)
            for name, a, b in self.events:
                key = name + "_cuda_span"
                result[key] = result.get(key, 0.) + a.elapsed_time(b) / 1000
        return result


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    return value


def cache_to_device(cache, device, step, max_age):
    age = step - cache["actor_step"]
    if not 0 <= age <= max_age:
        raise RuntimeError(f"OPD actor cache age {age} exceeds allowed 0..{max_age}")
    cpu_ids = cache["ids_cpu"]
    cache = tree_to(cache, device)
    cache["ids_cpu"] = cpu_ids  # text bucketing intentionally runs on CPU
    return cache


class RolloutClient:
    def __init__(self, server, group):
        self.server, self.group = server, group
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opd-actor-io")
        self.prepare_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opd-captions")
        self.pending = None
        self.snapshot_pending = None
        self.profile = False

    def request(self, payload):
        if self.pending is not None:
            raise RuntimeError("Only one future rollout batch per learner is allowed")
        # Payloads contain CPU caption IDs and strings, never mutable CUDA buffers.
        self.pending = self.pool.submit(self._request, payload)

    def request_prepared(self, prepare):
        """One CPU-only caption job; overlaps training and snapshot transport."""
        if self.pending is not None:
            raise RuntimeError("Only one future rollout batch per learner is allowed")
        prepared = self.prepare_pool.submit(prepare)
        self.pending = self.pool.submit(lambda: self._request(prepared.result()))

    def _request(self, payload):
        if self.snapshot_pending is not None:
            self.snapshot_pending.result()  # surface an install/ACK failure before requesting work
        encoded = "teacher_text" in payload
        dist.send(torch.tensor([3 if encoded else 1, 0]), self.server, group=self.group, tag=HEADER)
        if encoded:
            from aurora.opd_text import send_cpu_payload
            send_cpu_payload(payload, self.server, self.group)
        else:
            dist.send_object_list([payload], self.server, group=self.group)
        result = [None]
        dist.recv_object_list(result, self.server, group=self.group)
        return result[0]

    def collect(self):
        if self.pending is None:
            raise RuntimeError("No rollout batch requested")
        result = self.pending.result()
        self.pending = None
        return result

    def update(self, named, step):
        """Rank zero streams a consistent snapshot between optimizer updates.

        Use original parameter dtype: transport must not quantize the actor's
        policy. Streaming one parameter at a time avoids another full CPU model.
        """
        start = time.perf_counter()
        dist.send(torch.tensor([2, step]), self.server, group=self.group, tag=HEADER)
        for _, _, p in named:
            dist.send(p.detach().cpu().contiguous(), self.server, group=self.group, tag=WEIGHTS)
        ack = torch.empty(1, dtype=torch.long)
        dist.recv(ack, self.server, group=self.group, tag=ACK)
        if int(ack[0]) != step:
            raise RuntimeError("Rollout snapshot acknowledgement mismatch")
        if self.profile:
            print(f"actor snapshot v{step} -> rank {self.server}: transfer/install/ACK "
                  f"{time.perf_counter()-start:.2f}s (background)", flush=True)

    def update_async(self, named, step):
        """Freeze CPU storage before returning; the worker never reads live parameters.

        Snapshot capture intentionally blocks the main thread until D2H completes.
        Only transport/install/ACK overlap subsequent optimizer updates.
        """
        self.send_snapshot_async(capture_snapshot(named), step)

    def send_snapshot_async(self, snapshot, step):
        """Enqueue an already immutable CPU snapshot (possibly shared across actors)."""
        if self.snapshot_pending is not None:
            if not self.snapshot_pending.done():
                raise RuntimeError("Only one outstanding actor snapshot is allowed")
            self.snapshot_pending.result()
        if self.pending is not None:
            raise RuntimeError("Collect the previous rollout before updating the actor")
        self.snapshot_pending = self.pool.submit(self.update, snapshot, step)

    def stop(self):
        if self.pending is not None:
            self.collect()
        if self.snapshot_pending is not None:
            self.snapshot_pending.result()
        self.pool.shutdown(wait=True)
        self.prepare_pool.shutdown(wait=True)
        dist.send(torch.tensor([0, 0]), self.server, group=self.group, tag=HEADER)


def serve_rollouts(named, clients, group, generate):
    """Single GPU owner; generation never races with a weight replacement."""
    active, version = set(clients), None
    queued = {}
    while active:
        # Other learners can submit before rank zero's snapshot arrives. Receive
        # their payloads but never generate with an older policy than requested.
        ready = next((src for src, job in queued.items()
                      if version is not None and job.get("actor_step", version) == version), None)
        if ready is not None:
            payload = queued.pop(ready)
            started = time.perf_counter()
            with torch.no_grad():
                cache = generate(payload)
            cache["actor_step"] = version
            generated = time.perf_counter()
            cache = tree_to(cache, "cpu")
            copied = time.perf_counter()
            dist.send_object_list([cache], ready, group=group)
            sent = time.perf_counter()
            if "actor_timings" in cache:
                phases = " ".join(f"{k}={v:.2f}s" for k, v in cache["actor_timings"].items())
                print(f"actor rank {dist.get_rank()} -> learner {ready}, v{version}, "
                      f"{cache['resolution']}px {len(cache['ids'])} images: {phases} "
                      f"generate_total={generated-started:.2f}s cpu_copy={copied-generated:.2f}s "
                      f"send={sent-copied:.2f}s total={sent-started:.2f}s", flush=True)
            del cache, payload
            continue
        hdr = torch.empty(2, dtype=torch.long)
        src = dist.recv(hdr, group=group, tag=HEADER)
        if src not in active:
            raise RuntimeError(f"Unexpected rollout client {src}")
        kind = int(hdr[0])
        if kind == 0:
            active.remove(src)
        elif kind == 2:
            if src != 0:
                raise RuntimeError("Only learner rank zero may refresh actor weights")
            if any(job.get("actor_step", int(hdr[1])) != int(hdr[1]) for job in queued.values()):
                raise RuntimeError("Snapshot would invalidate queued rollout requests")
            with torch.no_grad():
                for _, _, p in named:
                    cpu = torch.empty(p.shape, dtype=p.dtype)
                    dist.recv(cpu, src, group=group, tag=WEIGHTS)
                    p.copy_(cpu)
            version = int(hdr[1])
            dist.send(torch.tensor([version]), src, group=group, tag=ACK)
        elif kind in (1, 3):
            if kind == 3:
                from aurora.opd_text import recv_cpu_payload
                payload = [recv_cpu_payload(src, group)]
            else:
                payload = [None]
                dist.recv_object_list(payload, src, group=group)
            requested = payload[0].get("actor_step", version)
            if requested is None:
                raise RuntimeError("Actor must receive initial learner snapshot before requests")
            if version is not None and requested < version:
                raise RuntimeError("Requested actor snapshot is already obsolete")
            if src in queued:
                raise RuntimeError("Only one queued rollout per learner is allowed")
            queued[src] = payload[0]
        else:
            raise RuntimeError(f"Unknown rollout request {kind}")
