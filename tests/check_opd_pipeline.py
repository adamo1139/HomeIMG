"""CPU checks of pipelined OPD gradients and teacher output ordering."""
import ast
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]

def extract(src, names):
    ns = dict(torch=torch, np=np, F=F, KIND_TEACHER=2, TAG_HDR=21, TAG_HW=24, TAG_IDS=22, TAG_MASK=23)
    nodes = [n for n in ast.parse(src).body if getattr(n, 'name', '') in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<test>', 'exec'), ns)
    return ns

class Stage(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.17))
        self.lm_states, self.text_cache, self._mb = lm, None, 0
    def te(self, ids, lm_states=None):
        return ids.float().mean(1) + (0 if lm_states is None else lm_states.mean((1,2))), None
    def forward(self, z, t, ids):
        text = self.text_cache if self.text_cache is not None else self.te(
            ids, None if self.lm_states is None else self.lm_states[self._mb])
        return self.weight*z.sin() + t[:,None,None,None] + .01*text[0][:,None,None,None]

class Teacher:
    def __init__(self, events):
        self.events, self.inputs = events, []
    def request(self, ids, mask, x, t, cap):
        self.events.append('request')
        self.inputs.append((x.clone(),t.clone()))
        return .2*x.cos()+.3*t[:,None,None,None]
    def collect(self,h):
        self.events.append('collect')
        return h

# Reference is the pre-interleaving implementation, pinned independently of HEAD.
old = subprocess.check_output(['git','show','8e3557f:train_pp.py'],cwd=ROOT,text=True)
new = (ROOT/'train_pp.py').read_text()
fake = ModuleType('aurora.klein')
fake.tokenize = lambda tok,texts: (torch.zeros(len(texts),2),torch.ones(len(texts),2))
for parts in (1,2,4):
    for m in (1,4):
        for use_lm in (False,True):
            results=[]
            for source in (old,new):
                torch.manual_seed(5)
                ns=extract(source,{'opd_step','opd_rollout','local_train_step'})
                lm=[torch.randn(2,3,2) for _ in range(8)] if use_lm else None
                stage=Stage(lm)
                events=[]
                stage.weight.register_hook(lambda g: events.append('backward'))
                teacher=Teacher(events)
                args=SimpleNamespace(opd_states=m,opd_rollout_steps=16,seed=7,dance_opd=m==1,
                                     dance_beta=(1,1),t_shift=6,text_bucket=False)
                losses=[]
                with patch.dict(sys.modules,{'aurora.klein':fake}):
                    ns['opd_step'](stage,teacher,None,['x']*16,torch.randn(16,2,2,2),
                                   torch.ones(16,3,dtype=torch.long),None,2,args,3,1,losses,parts)
                assert stage.lm_states is lm and stage.text_cache is None
                results.append((stage.weight.grad,torch.stack(losses),teacher.inputs,events))
            a,b=results
            for x,y in zip(a[:2],b[:2]): torch.testing.assert_close(x,y,rtol=0,atol=0)
            for pair1,pair2 in zip(a[2],b[2]):
                for x,y in zip(pair1,pair2): torch.testing.assert_close(x,y,rtol=0,atol=0)
            if parts==4:
                assert b[3].index('backward') < [i for i,e in enumerate(b[3]) if e=='request'][2]
print('24 configurations: identical states, losses and gradients; interleaving verified')

Client=extract((ROOT/'aurora/klein.py').read_text(),{'TeacherClient'})['TeacherClient']
class Work:
    def wait(self): pass
for nc in (2,8):
    for rank in (0,1,2):
        order=[]
        def send(buf,dst,**kw):
            if kw.get('tag') in (1,21): order.append(dst)
            return Work()
        def recv(buf,src):
            buf.fill_(src)
            return Work()
        with patch('torch.distributed.get_rank',return_value=rank), patch('torch.distributed.isend',side_effect=send), patch('torch.distributed.irecv',side_effect=recv):
            client=Client([3,4,5,6],'cpu',None,shared=[6])
            bounds=torch.linspace(0,nc,5).round().long().tolist()
            expected=torch.cat([torch.full(((hi-lo)*4,),float(s)) for s,lo,hi in zip([3,4,5,6],bounds[:-1],bounds[1:])])
            for call in range(4):
                order.clear()
                h=client.request(torch.ones(nc,512),torch.ones(nc,512),torch.zeros(nc*4,1,1,1),
                                 torch.ones(nc*4),torch.arange(nc).repeat_interleave(4))
                torch.testing.assert_close(client.collect(h)[:,0,0,0],expected)
                servers=[3,4,5,6]; offset=(rank+call)%4
                active={s for s,lo,hi in zip(servers,bounds[:-1],bounds[1:]) if hi>lo}
                assert order==[s for s in servers[offset:]+servers[:offset] if s in active]
print('24 dispatch configurations: rotated sends and original response order verified')
