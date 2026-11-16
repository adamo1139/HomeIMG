"""Bounded one-batch lookahead for a separate OPD rollout process.

Only CPU tensors travel on the dedicated Gloo group. A learner's I/O thread
receives the next cache while its main thread runs CUDA backward. CUDA work and
model snapshots remain on the main thread; no live model is shared with workers.
"""
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

HEADER, WEIGHTS, ACK = 310, 311, 312


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
        self.pending = None

    def request(self, payload):
        if self.pending is not None:
            raise RuntimeError("Only one future rollout batch per learner is allowed")
        # Payloads contain CPU caption IDs and strings, never mutable CUDA buffers.
        self.pending = self.pool.submit(self._request, payload)

    def _request(self, payload):
        dist.send(torch.tensor([1, 0]), self.server, group=self.group, tag=HEADER)
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
        dist.send(torch.tensor([2, step]), self.server, group=self.group, tag=HEADER)
        for _, _, p in named:
            dist.send(p.detach().cpu().contiguous(), self.server, group=self.group, tag=WEIGHTS)
        ack = torch.empty(1, dtype=torch.long)
        dist.recv(ack, self.server, group=self.group, tag=ACK)
        if int(ack[0]) != step:
            raise RuntimeError("Rollout snapshot acknowledgement mismatch")

    def stop(self):
        if self.pending is not None:
            self.collect()
        self.pool.shutdown(wait=True)
        dist.send(torch.tensor([0, 0]), self.server, group=self.group, tag=HEADER)


def serve_rollouts(named, clients, group, generate):
    """Single GPU owner; generation never races with a weight replacement."""
    active, version = set(clients), None
    while active:
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
            with torch.no_grad():
                for _, _, p in named:
                    cpu = torch.empty(p.shape, dtype=p.dtype)
                    dist.recv(cpu, src, group=group, tag=WEIGHTS)
                    p.copy_(cpu)
            version = int(hdr[1])
            dist.send(torch.tensor([version]), src, group=group, tag=ACK)
        elif kind == 1:
            payload = [None]
            dist.recv_object_list(payload, src, group=group)
            if version is None:
                raise RuntimeError("Actor must receive initial learner snapshot before requests")
            with torch.no_grad():
                cache = generate(payload[0])
            cache["actor_step"] = version
            cache = tree_to(cache, "cpu")
            dist.send_object_list([cache], src, group=group)
            del cache, payload
        else:
            raise RuntimeError(f"Unknown rollout request {kind}")
