"""Bounded CPU text-state lookahead, independent of actor policy snapshots."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

import torch
import torch.distributed as dist

TEXT_JOB = 4


def send_cpu_payload(payload, dst, group):
    """Send large states as tensors, without pickling copies of their storage."""
    tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    assert all(t.device.type == "cpu" for t in tensors.values())
    metadata = {k: v for k, v in payload.items() if k not in tensors}
    specs = [(k, tuple(t.shape), t.dtype) for k, t in tensors.items()]
    dist.send_object_list([(metadata, specs)], dst, group=group)
    for t in tensors.values():
        dist.send(t.contiguous(), dst, group=group)


def recv_cpu_payload(src, group):
    header = [None]
    dist.recv_object_list(header, src, group=group)
    metadata, specs = header[0]
    for key, shape, dtype in specs:
        t = torch.empty(shape, dtype=dtype)
        dist.recv(t, src, group=group)
        metadata[key] = t
    return metadata


class TextLookahead:
    def __init__(self, prepare, encode, start, stop, period, depth):
        assert period > 0 and depth > 0
        self.prepare, self.encode = prepare, encode
        self.next_step, self.stop_step, self.period, self.depth = start, stop, period, depth
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opd-text-lookahead")
        self.queue = deque()
        self._fill()

    def _job(self, step):
        # One worker owns the loader, caption cache and network connection.
        return self.encode(self.prepare(step))

    def _fill(self):
        while len(self.queue) < self.depth and self.next_step < self.stop_step:
            step = self.next_step
            self.queue.append((step, self.pool.submit(self._job, step)))
            self.next_step += self.period

    def take(self, step):
        if not self.queue or self.queue[0][0] != step:
            raise RuntimeError(f"Text queue out of order: requested update {step}")
        _, future = self.queue.popleft()
        self._fill()  # up to depth future batches, plus the batch being consumed
        return future.result()

    def close(self):
        # Drain network work before the learner sends the LM stop message.
        try:
            for _, future in self.queue:
                future.result()
        finally:
            self.pool.shutdown(wait=True)
            self.queue.clear()


def encode_remote(job, server, control_group, text_group):
    start = time.perf_counter()
    dist.send(torch.tensor([TEXT_JOB, 0, 0]), server, group=control_group, tag=1)
    send_cpu_payload(job, server, text_group)
    response = recv_cpu_payload(server, text_group)
    job.update(response)
    size = sum(t.numel() * t.element_size() for t in response.values() if isinstance(t, torch.Tensor))
    print(f"text queue learner {dist.get_rank()}: update {job['step']+1}, {len(job['ids'])} captions, "
          f"{size / 2**20:.0f} MiB CPU states, ready in {time.perf_counter()-start:.2f}s", flush=True)
    return job
