"""CPU regression for cached OPD: eight real updates, detached targets, fresh refiner graphs and data cadence."""
import ast
from types import SimpleNamespace
from unittest.mock import patch
import sys
import torch
from check_opd_pipeline import ROOT, extract, Stage, Teacher, fake

source = (ROOT/'train_pp.py').read_text()
ns = extract(source, {'opd_step','opd_replay','opd_rollout','local_train_step'})
class RefinerStage(Stage):
    def __init__(self,lm):
        super().__init__(lm)
        self.refiner = torch.nn.Parameter(torch.tensor(.7))
        self.encodes = 0
    def te(self, ids, lm_states=None):
        self.encodes += 1
        text,valid = super().te(ids,lm_states)
        return text*self.refiner,valid

def tensors(tree):
    if isinstance(tree,torch.Tensor): yield tree
    elif isinstance(tree,dict):
        for v in tree.values(): yield from tensors(v)
    elif isinstance(tree,(tuple,list)):
        for v in tree: yield from tensors(v)

for use_lm in (False,True):
    torch.manual_seed(3)
    lm=[torch.randn(2,3,2) for _ in range(8)] if use_lm else None
    stage=RefinerStage(lm)
    cache={}
    teacher=Teacher([])
    args=SimpleNamespace(opd_states=4,opd_rollout_steps=16,seed=7,dance_opd=False,t_shift=6,text_bucket=False)
    with patch.dict(sys.modules,{'aurora.klein':fake}):
        ns['opd_step'](stage,teacher,None,['x']*16,torch.randn(16,2,2,2),torch.ones(16,3,dtype=torch.long),
                       None,2,args,0,0,[],parts=4,cache=cache)
    assert len(teacher.inputs)==4
    for v in tensors(cache): assert v.grad_fn is None and not v.requires_grad
    originals=[v.clone() for v in tensors(cache)]
    ref=RefinerStage(None);ref.load_state_dict(stage.state_dict())
    opt=torch.optim.AdamW(stage.parameters(),lr=.001)
    refopt=torch.optim.AdamW(ref.parameters(),lr=.001)
    # Reference always recomputes forward/backward from fixed examples, independent of replay helper.
    for update in range(8):
        if update:
            encodes=stage.encodes
            ns['opd_replay'](stage,cache,2,[])
            assert stage.encodes>encodes
        count=0
        for b0,b1,x,t,target in cache['chunks']:
            full=cache['lm']
            if full is not None:
                full=full[b0:b1].repeat_interleave(4,dim=0)
                ref.lm_states=list(full.split(2))
            count+=ns['local_train_step'](ref,x,t,cache['ids'][b0:b1].repeat_interleave(4,dim=0),target,2,[],divide=False)
        for p in ref.parameters(): p.grad.div_(count)
        for p,q in zip(stage.parameters(),ref.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        opt.step();refopt.step()
        opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
        for p,q in zip(stage.parameters(),ref.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert stage.refiner.item()!=.7
    assert len(teacher.inputs)==4
    for old,v in zip(originals,tensors(cache)): torch.testing.assert_close(old,v,rtol=0,atol=0)
print('Replay: 8 AdamW updates match independent recomputation exactly, including trainable refiner')

# Execute the real data-preparation gate from main with a mocked loader.
main=next(n for n in ast.parse(source).body if getattr(n,'name','')=='main')
loop=next(n for n in main.body if isinstance(n,ast.While))
a=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='replay' for t in n.targets))
code=compile(ast.Module(body=loop.body[a:a+5],type_ignores=[]),'<main-data-gate>','exec')
for resume in (0,5,8):
    calls=[]
    def batch(step):
        calls.append(step)
        return torch.zeros(1),None,None,None,None,None,None,None
    env=dict(actor_client=None,args=SimpleNamespace(opd=True,opd_reuse=8),opd_next=None,opd_cache={},opd_pass=0,
             lm_client=None,next_batch=batch,stage_mod=SimpleNamespace(lm_states=None),lm_offload=False,
             torch=torch,opd_batches=3)
    for step in range(resume,resume+16):
        env['step']=step
        exec(code,env)
        if not env['replay']: env['opd_cache']['chunks']=['mock']
        env['opd_pass']+=1
    assert calls==[resume,resume+8],calls
    assert env['opd_batches']==5
print('Main data gate: fresh only once per 8 updates; fresh cache on resume, including mid-cycle')
