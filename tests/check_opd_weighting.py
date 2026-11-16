"""Bounded OPD priorities: bins, ties, detached weights and actual weighted replay gradients."""
from types import SimpleNamespace
from unittest.mock import patch
import sys
import torch
from check_opd_pipeline import ROOT, extract, Stage, Teacher, fake
ns=extract((ROOT/'train_pp.py').read_text(), {'opd_state_weights','opd_step','opd_replay','opd_rollout','local_train_step'})
weight=ns['opd_state_weights']
t=torch.tensor([.1]*4+[.3]*4+[.6]*4+[1.]*4)
e=torch.tensor([1.,2.,3.,4., 100.,200.,300.,400., 1.,1.,1.,2., 4.,4.,4.,4.],requires_grad=True)
w=weight(e,t)
assert not w.requires_grad and w.grad_fn is None
assert w.min()>=.5 and w.max()<=1.5
for group in w.split(4): torch.testing.assert_close(group.mean(),torch.tensor(1.))
torch.testing.assert_close(w[:4],torch.tensor([.5,.5,1.5,1.5]))
torch.testing.assert_close(w[:4],w[4:8])  # different scale in another bin cannot dominate
assert (w[8:]==1).all()  # tied median with no lower group, and all equal
assert (weight(torch.tensor([1.,2.,3.]),torch.zeros(3))==1).all()
assert (weight(torch.tensor([1.,2.,3.,float('nan')]),torch.zeros(4))==1).all()
# Odd groups retain mean 1 and do not arbitrarily split median ties.
wo=weight(torch.arange(5.),torch.zeros(5))
torch.testing.assert_close(wo,torch.tensor([.5,.5,1.,1.5,1.5]))
print('Weights: bounds, per-bin normalization, ties, sparse bins and stop-gradient verified')

class CountingStage(Stage):
    def __init__(self):
        super().__init__(None)
        self.calls=0
    def forward(self,*a):
        self.calls+=1
        return super().forward(*a)

torch.manual_seed(7)
stage=CountingStage(); cache={}; teacher=Teacher([])
args=SimpleNamespace(opd_states=4,opd_rollout_steps=16,seed=7,dance_opd=False,t_shift=6,
                     text_bucket=False,opd_state_weighting=True)
with patch.dict(sys.modules,{'aurora.klein':fake}):
    ns['opd_step'](stage,teacher,None,['x']*16,torch.randn(16,2,2,2),torch.ones(16,3,dtype=torch.long),
                   None,2,args,0,0,[],parts=4,cache=cache)
assert len(torch.cat(cache['errors']))==64
opt=torch.optim.SGD(stage.parameters(),lr=.01)
opt.step();opt.zero_grad(set_to_none=True)
# Independent reference, with same initial state and globally calculated lagged weights.
ref=CountingStage();ref.load_state_dict(stage.state_dict())
expected_weights=weight(torch.cat(cache['errors']),torch.cat([c[3] for c in cache['chunks']]))
assert (expected_weights!=1).any()
count=stage.calls
losses=[]
ns['opd_replay'](stage,cache,2,losses)
assert stage.calls-count==32  # one forward per training microbatch; no priority forward
assert len(teacher.inputs)==4
expected_errors=[]; expected_raw=[]; offset=0
for b0,b1,x,t,target in cache['chunks']:
    ids=cache['ids'][b0:b1].repeat_interleave(4,dim=0)
    for j in range(0,len(x),2):
        pred=ref(x[j:j+2],t[j:j+2],ids[j:j+2])
        err=(pred-target[j:j+2]).square().flatten(1).mean(1)
        expected_errors.append(err.detach());expected_raw.append(err.mean().detach())
        (err*expected_weights[offset+j:offset+j+2]).mean().backward()
    offset+=len(x)
for p in ref.parameters(): p.grad.div_(32)
for p,q in zip(stage.parameters(),ref.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
torch.testing.assert_close(torch.cat(cache['errors']),torch.cat(expected_errors),rtol=0,atol=0)
torch.testing.assert_close(torch.stack(losses),torch.stack(expected_raw),rtol=0,atol=0)
assert all(e.grad_fn is None for e in cache['errors'])
print('Weighted replay matches independent gradients, logs raw MSE, refreshes errors without extra forwards')
