"""CPU regression: mixed-size DanceOPD, replay, local prefetch and actor generation."""
import ast
from contextlib import nullcontext
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import sys
import time

import torch
from check_opd_pipeline import ROOT, extract, Stage, Teacher, fake

sys.path.insert(0, str(ROOT))
from aurora.opd_resolution import batch_resolution, time_shift, validate
from aurora.opd_actor import tree_to, cache_to_device

torch.set_num_threads(2)
source = (ROOT / "train_pp.py").read_text()
ns = extract(source, {"opd_step", "opd_rollout", "opd_replay", "local_train_step",
                      "opd_collect_prepared", "rollout_actor_server"})
args = SimpleNamespace(opd=True, pp=1, multi_res=[], crop_res=[], repa_weight=0,
                       opd_resolutions=[256, 512, 1024], opd_res_weights=[], opd_res_shifts=[1.5, 3, 6],
                       res=1024, t_shift=6, t_shift_ref=0, seed=87462,
                       opd_reuse=12, opd_states=1, opd_rollout_steps=4, dance_opd=True, dance_beta=(1, 1),
                       text_bucket=False, opd_parts=1, opd_state_weighting=False, cfg_drop=0,
                       no_compile=True, masked_attention_backend="sdpa", teacher_on_lm=True, klein="fake")
validate(args)
for field, value in [("opd_res_shifts", [1.5]), ("opd_res_weights", [1, -1, 1]),
                     ("opd_resolutions", [256, 512, 512]), ("pp", 2), ("opd", False)]:
    bad = SimpleNamespace(**vars(args)); setattr(bad, field, value)
    try:
        validate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(field)
automatic = SimpleNamespace(**vars(args)); automatic.opd_res_shifts = []
assert [time_shift(automatic, r) for r in args.opd_resolutions] == [1.5, 3, 6]
# Schedule is reproducible regardless of prefetch ordering or resume point.
schedule = {s: batch_resolution(args, s) for s in range(0, 1200, 12)}
assert set(schedule.values()) == {256, 512, 1024}
assert all(batch_resolution(args, s) == schedule[s] for s in reversed(schedule))
steps = [next(s for s, r in schedule.items() if r == res) for res in args.opd_resolutions]
ids = torch.ones(2, 3, dtype=torch.long)

# Run real next_batch against a fake data loader: OPD must not encode real images.
main = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "main")
node = next(n for n in main.body if getattr(n, "name", "") == "next_batch")
env = dict(torch=torch, args=args, batch_resolution=batch_resolution, needs_data=True, has_cond=False,
           profile=SimpleNamespace(phase=lambda _: nullcontext()), device=torch.device("cpu"),
           per_pipe=2, den_cfg=SimpleNamespace(latent_channels=2, latent_size=64), mr_groups=[],
           replica=0, shift_t=lambda t, r: t, null_row=ids[:1], null_cpu=ids[:1], hf_lm=True)
exec(compile(ast.Module(body=[node], type_ignores=[]), "<next_batch>", "exec"), env)
for res, step in zip(args.opd_resolutions, steps):
    env["loader"] = iter([(torch.empty(2, 0), ids, torch.zeros(2, 2, dtype=torch.long))])
    batch = env["next_batch"](step)
    eps = batch[3]
    assert eps.shape == (2, 2, res // 16, res // 16)
    assert not batch[0].any()
    for dance, prepare in ((False, False), (False, True), (True, False), (True, True)):
        args.dance_opd = dance
        stage, teacher, cache = Stage(None), Teacher([]), {}
        with patch.dict(sys.modules, {"aurora.klein": fake}):
            ns["opd_step"](stage, teacher, None, ["a", "b"], eps, ids, ids, 1, args,
                           step, 0, [], parts=1, cache=cache, prepare_only=prepare)
        if prepare:
            ns["opd_collect_prepared"](teacher, cache)
        assert cache["resolution"] == res and cache["time_shift"] == res / 1024 * 6
        # Independently compare to fixed-resolution OPD with the corresponding shift.
        fixed = SimpleNamespace(**vars(args)); fixed.opd_resolutions = []; fixed.t_shift = res / 1024 * 6
        reference = Teacher([])
        with patch.dict(sys.modules, {"aurora.klein": fake}):
            ns["opd_step"](Stage(None), reference, None, ["a", "b"], eps, ids, ids, 1,
                           fixed, step, 0, [], parts=1)
        for actual, expected in zip(teacher.inputs, reference.inputs):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        calls = len(teacher.inputs)
        opt = torch.optim.SGD(stage.parameters(), lr=.001)
        original = cache["chunks"][0][2].clone()
        for _ in range(12):
            opt.zero_grad(set_to_none=True)
            ns["opd_replay"](stage, cache, 1, [])
            opt.step()
        assert len(teacher.inputs) == calls
        assert torch.equal(original, cache["chunks"][0][2])
        assert cache["resolution"] == res
print("PASS: all 3 sizes/shifts, fixed-resolution equivalence, 12 replays, local prefetch, validation and deterministic schedule")

# Actual actor generate() must produce the same initial noise and resolution as next_batch.
class LM:
    def request(self, ids): return ids
    def collect(self, ids): return ids.float().unsqueeze(-1).repeat(1, 1, 2)
    def stop(self): pass
lm = LM()
teacher = Teacher([]); teacher.stop = lambda: None
services = ModuleType("aurora.klein")
services.tokenize = fake.tokenize
services.tokenizer = lambda *a: None
services.TeacherClient = lambda *a, **kw: teacher
ns.update(time=time, LMClient=lambda *a: lm, CrossAttention=type("Cross", (), {}),
          TextAttention=type("Text", (), {}), dist=SimpleNamespace(get_rank=lambda: 5))
actor = Stage(None); actor.blocks = torch.nn.ModuleList([])
def serve(named, clients, group, generate):
    for res, step in zip(args.opd_resolutions, steps):
        job = dict(ids=ids, texts=["a", "b"], step=step, replica=0, mb=1)
        cache = generate(job)
        cache["actor_step"] = step
        cache = cache_to_device(tree_to(cache, "cpu"), "cpu", step + 11, 23)
        assert cache["resolution"] == res and cache["time_shift"] == res / 1024 * 6
        assert cache["chunks"][0][2].shape[-2:] == (res // 16, res // 16)
        env["loader"] = iter([(torch.empty(2, 0), ids, torch.zeros(2, 2, dtype=torch.long))])
        eps = env["next_batch"](step)[3]
        ref = Stage([lm.collect(ids)[i:i+1] for i in range(2)])
        expected = {}
        ns["opd_step"](ref, teacher, None, job["texts"], eps, ids, ids, 1, args, step, 0, [],
                       parts=1, cache=expected, prepare_only=True)
        ns["opd_collect_prepared"](teacher, expected)
        for a, b in zip(cache["chunks"][0][2:], expected["chunks"][0][2:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
with patch.dict(sys.modules, {"aurora.klein": services}), \
     patch("aurora.opd_actor.serve_rollouts", side_effect=serve):
    ns["rollout_actor_server"](actor, [], SimpleNamespace(lm_dim=2),
                               SimpleNamespace(latent_channels=2, latent_size=64), args,
                               6, [6], [0], torch.device("cpu"), None, None)
print("PASS: actor matches local preparation for all sizes; cache metadata survives transport")

# Real training Stage: dynamic RoPE and whole-image conditioning match a model
# constructed specifically for that resolution (small width, real 16/32/64 grids).
import math
from dataclasses import replace
import aurora.model as model
stage_ns = dict(vars(model), math=math)
stage_node = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "Stage")
exec(compile(ast.Module(body=[stage_node], type_ignores=[]), "<Stage>", "exec"), stage_ns)
cfg = model.DenoiserConfig(latent_channels=2, latent_size=64, dim=16, depth=1,
                          heads=2, mlp_hidden=32, text_dim=8, rope_ref=64, micro_cond=256)
den = model.Denoiser(cfg); den.init_weights()
with torch.no_grad():
    den.ada.bias.normal_()  # activate attention gates so RoPE differences would be visible
text = torch.randn(1, 3, 8); valid = torch.ones(1, 3, dtype=torch.bool)
stage = stage_ns["Stage"](den, cfg, 0, 1, True, True)
stage.text_cache = (text, valid)
for res in args.opd_resolutions:
    side = res // 16
    ref = model.Denoiser(replace(cfg, latent_size=side))
    ref.load_state_dict(den.state_dict(), strict=True)
    x, t = torch.randn(1, 2, side, side), torch.tensor([.7])
    with torch.no_grad():
        actual = stage(x, t, ids[:1])
        expected = ref(x, t, text, valid,
                       cond=model.default_micro_cond(256, res, 1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
print("PASS: real Stage RoPE and micro-conditioning match fixed-resolution denoisers")
