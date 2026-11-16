"""Exercise actual actor generation with tiny modules and in-memory LM/teacher services."""
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import sys
import time
import torch
from check_opd_pipeline import ROOT, Stage, Teacher, extract

sys.path.insert(0, str(ROOT))
src = (ROOT / 'train_pp.py').read_text()
ns = extract(src, {'rollout_actor_server', 'opd_step', 'opd_rollout', 'opd_collect_prepared'})
args = SimpleNamespace(no_compile=True, masked_attention_backend='sdpa', teacher_on_lm=True,
                       klein='fake', seed=7, opd_states=4, opd_rollout_steps=16, dance_opd=False,
                       t_shift=6, text_bucket=False, opd_parts=5, opd_state_weighting=True)
class LM:
    def __init__(self, *args): self.stopped = False
    def request(self, ids): return ids
    def collect(self, ids): return ids.float().unsqueeze(-1).repeat(1, 1, 2) * .3
    def stop(self): self.stopped = True
lm = LM()
teacher = Teacher([])
teacher.stop = lambda: None
fake = ModuleType('aurora.klein')
fake.TeacherClient = lambda *a, **kw: teacher
fake.tokenizer = lambda *a: None
fake.tokenize = lambda tok, texts: (torch.zeros(len(texts), 2), torch.ones(len(texts), 2))
job = dict(ids=torch.arange(30).view(10,3), texts=['x'] * 10, step=8, replica=2, mb=2)
actor = Stage(None)
actor.blocks = torch.nn.ModuleList([])
ns.update(time=time,LMClient=lambda *a: lm, CrossAttention=type('Cross', (), {}),
          TextAttention=type('Text', (), {}), dist=SimpleNamespace(get_rank=lambda: 5))
results = []
def serve(named, clients, group, generate):
    with torch.no_grad():
        results.append(generate(job))
        actor.weight.add_(.2)  # simulate a new snapshot between requests
        results.append(generate(job))
with patch.dict(sys.modules, {'aurora.klein': fake}), patch('aurora.opd_actor.serve_rollouts', side_effect=serve):
    ns['rollout_actor_server'](actor, [], SimpleNamespace(lm_dim=2),
                               SimpleNamespace(latent_channels=2, latent_size=2), args, 6, [6], [0,1,2,3,4],
                               torch.device('cpu'), None, None)
assert lm.stopped and actor.lm_states is None and actor.weight.grad is None
for version, actual in enumerate(results):
    reference = Stage(None)
    with torch.no_grad(): reference.weight.fill_(.17 + .2 * version)
    full = lm.collect(job['ids'])
    reference.lm_states = list(full.split(2))
    rng = torch.Generator().manual_seed(7 * 1_000_003 + 2 * 1009 + 8)
    torch.randn(10, generator=rng)
    eps = torch.randn(10,2,2,2, generator=rng)
    expected = {}
    t = Teacher([])
    with patch.dict(sys.modules, {'aurora.klein': fake}), torch.no_grad():
        ns['opd_step'](reference, t, None, job['texts'], eps, job['ids'], job['ids'], 2, args, 8, 2, [],
                       parts=5, cache=expected, prepare_only=True)
        ns['opd_collect_prepared'](t, expected)
    assert 'pending' not in actual and actual['errors'] == []
    for a,b in zip(actual['chunks'], expected['chunks']):
        assert a[:2] == b[:2]
        for x,y in zip(a[2:], b[2:]):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            assert not x.requires_grad
print('Actor generation matches local states/targets exactly across two policy snapshots; no gradients or LM cache leak')
