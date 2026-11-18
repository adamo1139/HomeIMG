"""Two Gloo actors must run concurrently on the same immutable snapshot."""
import datetime
import ast
import os
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aurora.opd_actor import RolloutClient, capture_snapshot, serve_rollouts


def check_layout():
    main = next(n for n in ast.parse(Path("train_pp.py").read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body) if ast.unparse(n).startswith("world = world_all"))
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "os.environ.get('PP_CPU')")
    code = compile(ast.Module(body=main.body[start:end], type_ignores=[]), "<rank-layout>", "exec")
    for gpus, expected in [([2, 7], [1, 3, 5, 6, 2, 7, 4]), ([], [1, 3, 5, 6, 7, 2, 4])]:
        args = SimpleNamespace(lm_rank=1, teacher_ranks=0, vae_rank=0, teacher_on_lm=True,
                               opd_actor=True, skip_gpus=[0], lm_gpu=4, vae_gpu=0,
                               opd_actor_gpus=gpus, opd_actor_gpu=-1 if gpus else 2)
        for rank in range(7):
            env = dict(args=args, world_all=7, rank=rank, pp=1, actor_count=len(gpus) or 1,
                       torch=torch, os=os)
            with patch.dict(os.environ, LOCAL_RANK=str(rank)), patch("torch.cuda.device_count", return_value=8):
                exec(code, env)
            assert env["local"] == expected[rank]
            assert env["teachers"] == [6]
            assert env["teacher_clients"] == ([4, 5] if gpus else [5])
    print("PASS: production rank layout, physical GPU mapping and teacher clients for one/two actors")


def worker(rank, path, started):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=45))
    group = dist.new_group(backend="gloo")
    learners = dist.new_group([0, 1], backend="gloo")
    p = torch.nn.Parameter(torch.zeros(3))
    named = [("den", "w", p)]
    if rank >= 2:
        def generate(job):
            assert job["replica"] == rank - 2
            started[rank - 2].set()
            # Neither actor can finish until both are generating: proves overlap.
            assert started[3 - rank].wait(30)
            return dict(ids=job["ids"], ids_cpu=job["ids"], values=p.clone())
        serve_rollouts(named, [0, 1], group, generate)
    else:
        connections = [RolloutClient(ar, group) for ar in (2, 3)]
        for version in (0, 8, 16):
            if rank == 0:
                with torch.no_grad():
                    p.fill_(version)
                snapshot = capture_snapshot(named)
                for client in connections:
                    client.send_snapshot_async(snapshot, version)
                with torch.no_grad():
                    p.fill_(-100)
            # Requests may reach the second actor before its weights do.
            client = connections[rank]
            client.request_prepared(lambda v=version: dict(ids=torch.ones(2, 3, dtype=torch.long),
                                                          replica=rank, actor_step=v))
            cache = client.collect()
            assert cache["actor_step"] == version
            torch.testing.assert_close(cache["values"], torch.full((3,), float(version)))
            dist.barrier(group=learners)
            if rank == 0:
                for event in started:
                    event.clear()
            dist.barrier(group=learners)
        for client in connections:
            client.stop()
    dist.destroy_process_group()


if __name__ == "__main__":
    check_layout()
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(str(Path(tmp) / "rendezvous"), [ctx.Event(), ctx.Event()]),
                 nprocs=4, join=True)
    print("PASS: two actors generate concurrently, three shared immutable snapshots, exact versions and clean shutdown")
