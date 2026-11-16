"""CPU/Gloo actor transport: real processes, async receive, snapshots, bounded queue and age checks."""
import ast
import datetime
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aurora.opd_actor import RolloutClient, cache_to_device, serve_rollouts


def worker(rank, path, started, release):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank, world_size=3,
                            timeout=datetime.timedelta(seconds=45))
    group = dist.new_group(backend='gloo')
    learners = dist.new_group([0, 1], backend='gloo')
    p = torch.nn.Parameter(torch.zeros(3))
    q = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    named = [('den', 'w', p), ('te', 'w', q)]
    if rank == 2:
        def generate(job):
            who = job['rank']
            started[who].set()
            assert release[who].wait(30)
            ids = job['ids']
            return dict(ids=ids, ids_cpu=ids, lm=q * 2, chunks=[(0, 1, p * 2, p + 1, p + 2)],
                        m=1, errors=[], weighting=True)
        serve_rollouts(named, [0, 1], group, generate)
    else:
        client = RolloutClient(2, group)
        for version in (3, 11):
            if rank == 0:
                with torch.no_grad():
                    p.fill_(version)
                    q.fill_(version + 1)
                client.update(named, version)
            dist.barrier(group=learners)
            client.request(dict(ids=torch.ones(1, 2, dtype=torch.long), rank=rank))
            try:
                client.request({})
                raise AssertionError('second outstanding request accepted')
            except RuntimeError:
                pass
            assert started[rank].wait(30)
            assert not client.pending.done()  # generation is blocked, learner thread is free
            release[rank].set()
            cache = client.collect()
            assert cache['actor_step'] == version
            torch.testing.assert_close(cache['chunks'][0][2], torch.full((3,), 2. * version))
            torch.testing.assert_close(cache['lm'], torch.full((2,), 2. * (version + 1), dtype=torch.bfloat16))
            for step in (version, version + 8):
                converted = cache_to_device(cache, 'cpu', step, 8)
                assert not converted['chunks'][0][2].requires_grad
                assert converted['ids_cpu'].device.type == 'cpu'
            for step in (version - 1, version + 9):
                try:
                    cache_to_device(cache, 'cpu', step, 8)
                    raise AssertionError('invalid age accepted')
                except RuntimeError:
                    pass
            started[rank].clear()
            release[rank].clear()
            dist.barrier(group=learners)
        client.stop()
    dist.destroy_process_group()


def check_schedule():
    src = (ROOT / 'train_pp.py').read_text()
    main = next(n for n in ast.parse(src).body if getattr(n, 'name', '') == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.While))
    a = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'replay' for t in n.targets))
    gate = compile(ast.Module(body=loop.body[a:a+5], type_ignores=[]), '<actor gate>', 'exec')
    prefetch = next(n for n in loop.body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'actor_client is not None')
    prefetch = compile(ast.Module(body=[prefetch], type_ignores=[]), '<actor lookahead>', 'exec')
    for start, end in ((0, 32), (5, 37), (8, 21), (9, 10)):
        events, refreshes, consumed = [], [], []
        client = SimpleNamespace(pending=None)
        env = dict(actor_client=client, args=SimpleNamespace(opd=True, opd_reuse=8, steps=end),
                   opd_cache={}, opd_next=None, opd_pass=0, opd_batches=0, device='cpu', time=time)
        def refresh():
            env['version'] = env['step']
            refreshes.append(env['step'])
        def request(at):
            assert client.pending is None
            events.append(at)
            client.pending = dict(actor_step=env['version'], ids_cpu=torch.ones(1), chunks=[])
        def collect():
            assert client.pending is not None
            result, client.pending = client.pending, None
            consumed.append(env['step'])
            return result
        client.collect = collect
        env.update(refresh_actor=refresh, request_actor=request, cache_to_device=cache_to_device)
        ages = []
        for step in range(start, end):
            env['step'] = step
            exec(gate, env)
            exec(prefetch, env)
            ages.append(step - env['opd_cache']['actor_step'])
            env['opd_pass'] += 1
        assert events == consumed == list(range(start, end, 8)), (events, consumed)
        assert env['opd_batches'] == len(consumed) and client.pending is None
        assert max(ages) <= 15
        assert refreshes == [start] + list(range(start, end - 8, 8))
        if end - start >= 16:
            assert ages[8:16] == list(range(8,16))
    print('Real main scheduling: reuse 8, initial/resumed snapshot, fresh cycle snapshots, full-cycle lookahead, <=15-update state age, no skipped/excess batches')


if __name__ == '__main__':
    check_schedule()
    ctx = mp.get_context('spawn')
    started = [ctx.Event(), ctx.Event()]
    release = [ctx.Event(), ctx.Event()]
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(str(Path(tmp) / 'rendezvous'), started, release), nprocs=3, join=True)
    print('3 real Gloo processes: asynchronous cache receipt, two exact snapshots, queue bound, age rejection and shutdown passed')
