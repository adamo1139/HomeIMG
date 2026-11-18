"""Bounded lookahead and real CPU/Gloo preencoded-text teacher protocol."""
import datetime
import tempfile
from pathlib import Path
from types import SimpleNamespace
from threading import Event
from unittest.mock import patch
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aurora.opd_text import TextLookahead, encode_remote
from aurora.klein import TeacherClient, serve_request, TAG_HDR
from train_pp import lm_server, LMClient
from check_lm_server import make_lm


def bounded():
    calls, filled = [], Event()
    def prepare(step):
        calls.append(step)
        if len(calls) == 4:
            filled.set()
        return dict(step=step)
    q = TextLookahead(prepare, lambda job: job, 13, 54, 8, 4)
    assert filled.wait(10)
    assert calls == [13, 21, 29, 37]  # resume starts from consumed data, no preceding jobs
    assert len(q.queue) == 4
    for step in (13, 21, 29, 37, 45, 53):
        assert q.take(step)["step"] == step
        assert len(q.queue) <= 4
    q.close()
    assert calls == [13, 21, 29, 37, 45, 53]
    print("PASS: four queued batches, ordered consumption, resume and no work beyond final update")


class FakeQwen:
    tok = None
    def encode(self, prompts):
        return torch.zeros(1, 512, 7680, dtype=torch.bfloat16)
    def encode_ids(self, ids, mask):
        return ids[:, :1, None].expand(-1, 512, 7680).to(torch.bfloat16).contiguous()


class FakeTeacher:
    dtype = torch.bfloat16
    def encode_ids(self, *args):
        raise AssertionError("Teacher must not run a text encoder")
    def velocity(self, x, t, text, null, cfg):
        assert null.count_nonzero() == 0
        return x + text[:, 0, 0, None, None, None] + cfg


def worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=90))
    control = dist.new_group(backend='gloo')
    text_group = dist.new_group(backend='gloo')
    device = torch.device('cpu')
    if rank == 3:
        def tokenize(tok, texts):
            ids = torch.tensor([int(t) for t in texts])[:, None].expand(-1, 512).contiguous()
            return ids, torch.ones_like(ids)
        with patch('aurora.klein.tokenize', side_effect=tokenize):
            lm_server(SimpleNamespace(repo='fake', lm_dim=8), [0, 1], device, print, control,
                      lm=make_lm(), text_encoder=FakeQwen(), text_group=text_group)
    elif rank == 2:
        active = {0, 1}
        while active:
            hdr = torch.empty(3, dtype=torch.long)
            src = dist.recv(hdr, group=control, tag=TAG_HDR)
            if int(hdr[0]) == 0:
                active.remove(src)
            else:
                serve_request(FakeTeacher(), None, src, hdr, control, device, 5)
    else:
        def prepare(step):
            return dict(step=step, ids=torch.ones(2, 64, dtype=torch.long),
                        texts=[str(step + rank + 1)] * 2)
        queue = TextLookahead(prepare, lambda job: encode_remote(job, 3, control, text_group), 0, 17, 8, 4)
        teacher = TeacherClient([2], device, control)
        for step in (0, 8, 16):
            job = queue.take(step)
            assert job['lm'].shape == (2, 64, 8)
            x = torch.zeros(2, 128, 1, 1)
            h = teacher.request(job['q_ids'], job['q_mask'], x, torch.ones(2), torch.arange(2),
                                text=job['teacher_text'], null=job['teacher_null'])
            torch.testing.assert_close(teacher.collect(h), torch.full_like(x, step + rank + 6))
        queue.close()
        teacher.stop()
        LMClient(3, 8, device, control).stop()
    dist.destroy_process_group()


if __name__ == '__main__':
    bounded()
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(str(Path(tmp) / 'rendezvous'),), nprocs=4, join=True)
    print('PASS: both text encoders, CPU queue, preencoded teacher targets, concurrent clients and shutdown')
