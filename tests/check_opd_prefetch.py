"""CPU regression of deferred teacher collection and one-cache lookahead scheduling."""
import ast
from types import SimpleNamespace
from unittest.mock import patch
import sys
import torch
from check_opd_pipeline import ROOT, extract, Stage, Teacher, fake
src=(ROOT/'train_pp.py').read_text()
ns=extract(src,{'opd_step','opd_rollout','local_train_step','opd_replay','opd_state_weights','opd_collect_prepared'})
for weighted in (False,True):
    for use_lm in (False,True):
        caches=[]; stages=[]; teachers=[]
        args=SimpleNamespace(opd_states=4,opd_rollout_steps=16,seed=7,dance_opd=False,t_shift=6,
                             text_bucket=False,opd_state_weighting=weighted)
        for prepare in (False,True):
            torch.manual_seed(9)
            lm=[torch.randn(2,3,2) for _ in range(4)] if use_lm else None
            stage=Stage(lm); teacher=Teacher([]);cache={}
            with patch.dict(sys.modules,{'aurora.klein':fake}):
                ns['opd_step'](stage,teacher,None,['x']*8,torch.randn(8,2,2,2),torch.ones(8,3,dtype=torch.long),
                               None,2,args,8,0,[],parts=4,cache=cache,prepare_only=prepare)
            if prepare:
                assert stage.weight.grad is None
                assert teacher.events==['request']*4
                assert len(cache['pending'])==4 and not cache['chunks']
            else:
                stage.zero_grad(set_to_none=True)
            # Simulate intervening learner updates before the prepared states are first trained on.
            with torch.no_grad(): stage.weight.add_(.13)
            if prepare:
                ns['opd_collect_prepared'](teacher,cache)
                assert 'pending' not in cache
            cache['errors']=[]  # first use should be unweighted even when weighting is enabled
            losses=[]
            ns['opd_replay'](stage,cache,2,losses)
            assert stage.lm_states is lm and stage.text_cache is None
            stages.append((stage.weight.grad,torch.stack(losses)))
            caches.append(cache);teachers.append(teacher)
        for x,y in zip(*stages): torch.testing.assert_close(x,y,rtol=0,atol=0)
        for c1,c2 in zip(caches[0]['chunks'],caches[1]['chunks']):
            for x,y in zip(c1[2:],c2[2:]): torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert teachers[1].events==['request']*4+['collect']*4
print('Prepared cache: no backward/collection during generation; exact gradients after intervening weight changes')

main=next(n for n in ast.parse(src).body if getattr(n,'name','')=='main')
loop=next(n for n in main.body if isinstance(n,ast.While))
a=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='replay' for t in n.targets))
gate=compile(ast.Module(body=loop.body[a:a+4],type_ignores=[]),'<gate>','exec')
prefetch=next(n for n in loop.body if isinstance(n,ast.If) and 'opd_prefetch' in ast.unparse(n.test))
condition=compile(ast.Expression(prefetch.test),'<prefetch-condition>','eval')
for start in (0,5,8):
    calls=[]
    def batch(step):
        calls.append(step)
        return torch.zeros(1),None,None,None,None,None,None,None
    env=dict(args=SimpleNamespace(opd=True,opd_reuse=8,opd_prefetch=True,steps=start+32),
             opd_next=None,opd_cache={},opd_pass=0,opd_batches=0,lm_client=None,next_batch=batch,
             stage_mod=SimpleNamespace(lm_states=None),lm_offload=False,torch=torch)
    activations=[];prefetches=[]
    for step in range(start,start+32):
        env['step']=step
        exec(gate,env)
        if not env['replay']:
            activations.append(step)
            env['opd_cache']['chunks']=['current']
        if eval(condition,env):
            future=step+8-env['opd_pass']
            batch(future)
            env['opd_next']={'chunks':['future']}
            prefetches.append(step)
        env['opd_pass']+=1
    assert calls==[start,start+8,start+16,start+24],calls
    assert activations==calls and prefetches==[start+1,start+9,start+17]
    assert env['opd_batches']==4 and env['opd_next'] is None
print('32-update scheduling: four batches, one lookahead, no unused final prefetch, resume cadence verified')
