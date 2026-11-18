"""Token-balanced batches: real data preparation, actor, replay gradients and resume position."""
import numpy as np
from check_opd_resolution import *
from aurora.opd_resolution import batch_multiplier, gather_caption_batches, microbatch_size, validate_microbatches
from aurora.data import TrainStream

args.opd_token_batch = True
args.opd_res_mb = [16, 8, 2]
validate(args)
validate_microbatches(SimpleNamespace(**{**vars(args), "opd_parts": 5}), 10, 2)
try:
    validate_microbatches(SimpleNamespace(**{**vars(args), "opd_parts": 5, "opd_res_mb": [32, 16, 2]}), 10, 2)
except ValueError:
    pass
else:
    raise AssertionError("Partial rollout microbatches must be rejected")
assert [50 * batch_multiplier(args, r) for r in args.opd_resolutions] == [800, 200, 50]
assert {50 * batch_multiplier(args, r) * (r // 16)**2 for r in args.opd_resolutions} == {204800}

class Store:
    def __len__(self): return 10000
    def fetch(self, indices, rng, rows=True):
        return (np.empty((len(indices), 0), dtype=np.float32),
                np.repeat(indices[:, None], 3, axis=1), np.stack([indices, indices], axis=1))

def stream(start=0):
    return iter(TrainStream(Store(), 2, 0, 5, args.seed, start_micro=start, rows=True))

load_node = next(n for n in main.body if getattr(n, "name", "") == "load_rollout_batch")
env.update(batch_multiplier=batch_multiplier, gather_caption_batches=gather_caption_batches)
exec(compile(ast.Module(body=[load_node], type_ignores=[]), "<load_rollout_batch>", "exec"), env)

consumed = 0
env["loader"] = stream()
for res, step in zip(args.opd_resolutions, steps):
    factor = batch_multiplier(args, res)
    nb = env["next_batch"](step)
    consumed += factor
    assert nb[3].shape == (2 * factor, 2, res // 16, res // 16)
    assert nb[1].shape[0] == 2 * factor
    local = Stage(None); teacher = Teacher([]); cache = {}
    mb_res = microbatch_size(args, res, 1)
    with patch.dict(sys.modules, {"aurora.klein": fake}):
        ns["opd_step"](local, teacher, None, ["x"] * len(nb[1]), nb[3], nb[4], nb[5],
                       mb_res, args, step, 0, [], parts=1, cache=cache)
    # Global mean over all states, independent of accumulated microbatch count.
    reference = Stage(None)
    b0, b1, x, t, target = cache["chunks"][0]
    ((reference(x, t, cache["ids"]) - target)**2).mean().backward()
    torch.testing.assert_close(local.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-5)
    local.zero_grad(set_to_none=True)
    ns["opd_replay"](local, cache, mb_res, [])
    torch.testing.assert_close(local.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-5)

# A fetched future batch is discarded on resume: resume uses consumed base batches,
# not number of rollout batches or the loader's speculative cursor.
future = gather_caption_batches(env["loader"], 16)
resumed = gather_caption_batches(stream(consumed), 16)
assert torch.equal(future[1], resumed[1]) and torch.equal(future[2], resumed[2])
assert consumed == 21
# Run the production consumed-data accounting for fresh, replay and actor-prefetched caches.
loop = next(n for n in main.body if isinstance(n, ast.While))
accounting = next(n for n in loop.body if isinstance(n, ast.If)
                  and "opd_data_batches +=" in ast.unparse(n))
code = compile(ast.Module(body=[accounting], type_ignores=[]), "<consumed-data>", "exec")
account_env = dict(args=args, batch_multiplier=batch_multiplier, opd_data_batches=0)
for res in args.opd_resolutions:
    for reuse_pass in range(12):
        account_env.update(active_opd_res=res, replay=reuse_pass > 0)
        exec(code, account_env)
assert account_env["opd_data_batches"] == consumed
print("PASS: 800/200/50 global images, equal image tokens, complete captions, gradient normalization and resume after lookahead")

# Exercise real request_actor + generate with enlarged batches and per-resolution microbatches.
request_node = next(n for n in main.body if getattr(n, "name", "") == "request_actor")
jobs = []
preparations = []
env.update(actor_client=SimpleNamespace(request_prepared=preparations.append),
           captions_en=lambda rows: ["x"] * len(rows), mb=99, base_mb=1,
           microbatch_size=microbatch_size, actor_snapshot_step=0, time=time, text_lookahead=None)
exec(compile(ast.Module(body=[request_node], type_ignores=[]), "<request_actor>", "exec"), env)
env["loader"] = stream()
for step in steps:
    env["request_actor"](step)
env["actor_snapshot_step"] = 999
jobs = [prepare() for prepare in preparations]
assert all(job["actor_step"] == 0 for job in jobs)

actor = Stage(None); actor.blocks = torch.nn.ModuleList([])
teacher = Teacher([]); teacher.stop = lambda: None
services.TeacherClient = lambda *a, **kw: teacher
def serve_balanced(named, clients, group, generate):
    for res, job in zip(args.opd_resolutions, jobs):
        n = 2 * batch_multiplier(args, res)
        mb_res = microbatch_size(args, res, 1)
        assert len(job["ids"]) == len(job["texts"]) == n and job["mb"] == mb_res
        cache = generate(job)
        # Frozen text states from the queue must produce exactly the same rollout/targets.
        q_ids, q_mask = fake.tokenize(None, job["texts"])
        cached_job = dict(job, lm=lm.collect(job["ids"]), q_ids=q_ids, q_mask=q_mask,
                          teacher_text=torch.zeros(n, 2, 2), teacher_null=torch.zeros(1, 2, 2))
        original_request = teacher.request
        def preencoded_request(ids, mask, x, t, cap, text=None, null=None):
            assert text is not None and len(text) == len(ids) and null is not None
            return original_request(ids, mask, x, t, cap)
        with patch.object(teacher, "request", side_effect=preencoded_request), \
             patch.object(lm, "request", side_effect=AssertionError("Unexpected online LM request")):
            cached = generate(cached_job)
        for a, b in zip(cache["chunks"], cached["chunks"]):
            for x, y in zip(a[2:], b[2:]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert all(value >= 0 for value in cache["actor_timings"].values())
        assert {"lm_wall", "rollout_wall", "teacher_request_wall", "teacher_collect_wall"} <= cache["actor_timings"].keys()
        assert len(cache["ids"]) == n and cache["resolution"] == res
        assert sum(len(chunk[2]) for chunk in cache["chunks"]) == n
        learner = Stage(None)
        ns["opd_replay"](learner, cache, mb_res, [])
        assert torch.isfinite(learner.weight.grad)
args.opd_actor_profile = True
with patch.dict(sys.modules, {"aurora.klein": services}), \
     patch("aurora.opd_actor.serve_rollouts", side_effect=serve_balanced):
    ns["rollout_actor_server"](actor, [], SimpleNamespace(lm_dim=2),
                               SimpleNamespace(latent_channels=2, latent_size=64), args,
                               6, [6], [0], torch.device("cpu"), None, None)
print("PASS: enlarged batches through actual actor request, generation, LM cache and replay")

# The current cache controls replay, even when the next job has another resolution.
selection = next(n for n in loop.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "args.opd" and "microbatch_size" in ast.unparse(n))
select_code = compile(ast.Module(body=[selection], type_ignores=[]), "<microbatch-selection>", "exec")
select_env = dict(args=args, microbatch_size=microbatch_size, batch_resolution=batch_resolution,
                  base_mb=2, step=steps[2], opd_cache={"resolution": 256}, opd_next={"resolution": 512})
for replay, prepared, expected in [(True, False, 16), (False, True, 8), (False, False, 2)]:
    select_env.update(replay=replay, prepared=prepared)
    exec(select_code, select_env)
    assert select_env["mb"] == expected
print("PASS: replay, local prefetch and fresh actor jobs select their own microbatch")
