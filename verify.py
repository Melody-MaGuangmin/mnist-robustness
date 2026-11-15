"""Reload every checkpoint and regenerate all test inputs, predictions and metrics."""
from train import (ROOT,PARAMS,read_protocol,sha256,write_json,split_indices,load_training,
                   load_testing,condition_batches,evaluate_condition,initialize,loss_grad,shift_images)
import json,time
import numpy as np

def unit_checks(cfg):
    a=np.zeros((1,784),dtype=np.float64);a[0,0]=1
    assert shift_images(a,-2,0).sum()==0
    assert shift_images(a,0,2)[0,2]==1
    p={k:v.astype(np.float64) for k,v in initialize(7,cfg).items()}
    rng=np.random.default_rng(1);x=rng.uniform(.1,.9,(4,784));y=np.array([0,2,4,6])
    _,g=loss_grad(p,x,y,cfg['optimizer']['l2_coefficient']);errors=[]
    for name in PARAMS:
        for flat in rng.choice(p[name].size,10,replace=False):
            idx=np.unravel_index(flat,p[name].shape);old=p[name][idx];eps=1e-5
            p[name][idx]=old+eps;lp=loss_grad(p,x,y,cfg['optimizer']['l2_coefficient'])[0]
            p[name][idx]=old-eps;lm=loss_grad(p,x,y,cfg['optimizer']['l2_coefficient'])[0]
            p[name][idx]=old;err=abs((lp-lm)/(2*eps)-g[name][idx]);errors.append(err)
            assert err<1e-6,(name,idx,err)
    return max(errors)

def main():
    start=time.perf_counter();cfg=read_protocol();grad_error=unit_checks(cfg)
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest,rel=line.split('  ',1);assert sha256(ROOT/rel)==digest,rel
    x,y,tr,va=load_training(ROOT,cfg)
    with np.load(ROOT/'data/split_indices.npz',allow_pickle=False) as f:
        assert np.array_equal(f['train_idx'],tr) and np.array_equal(f['validation_idx'],va)
    assert len(tr)==55000 and len(va)==5000 and not set(tr)&set(va)
    assert np.array_equal(np.bincount(y[va],minlength=10),np.full(10,500))
    # Verify random streams are invariant to generation batch size.
    for c in cfg['test']['conditions']:
        aa=np.concatenate([z for z,_ in condition_batches(x[:37],y[:37],c,'test',cfg,batch_size=7)])
        bb=np.concatenate([z for z,_ in condition_batches(x[:37],y[:37],c,'test',cfg,batch_size=19)])
        assert np.array_equal(aa,bb),c
    del x,y
    tx,ty=load_testing(ROOT,cfg)
    metrics=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))
    indexed={(r['method'],r['seed'],r['condition']):r for r in metrics}
    expected_hashes=json.loads((ROOT/'results/test_input_hashes.json').read_text(encoding='utf-8'))
    count=0
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            stem=f'{method}_seed{seed}'
            with np.load(ROOT/f'weights/{stem}.npz',allow_pickle=False) as f:p={k:f[k] for k in PARAMS}
            with np.load(ROOT/f'results/predictions/{stem}.npz',allow_pickle=False) as ps,np.load(ROOT/f'results/confusions/{stem}.npz',allow_pickle=False) as cs:
                for c in cfg['test']['conditions']:
                    pred,cm,m,hashes=evaluate_condition(p,tx,ty,c,cfg)
                    assert np.array_equal(pred,ps[c]),(stem,c,'predictions')
                    assert np.array_equal(cm,cs[c]),(stem,c,'confusion')
                    assert hashes==expected_hashes[c],(stem,c,'input hashes')
                    expected=indexed[method,seed,c]
                    for key in m:assert abs(m[key]-expected[key])<1e-12,(stem,c,key)
                    count+=1
            print(json.dumps({'verified_checkpoint':stem}),flush=True)
    assert count==140
    report={'status':'PASS','reloaded_checkpoints':20,'metric_rows_reproduced':count,'predictions':'exact match',
            'regenerated_test_input_hashes':'all match','artifact_hashes':'all match','split_disjoint':True,
            'validation_class_balance':'500 per digit','corruption_generation_batch_size_invariant':True,
            'finite_difference_checks':40,'max_gradient_absolute_error':grad_error,'wall_seconds':time.perf_counter()-start}
    write_json(ROOT/'verification.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
