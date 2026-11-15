"""Real MNIST experiments using only NumPy. Protocol is fixed before evaluation."""
import os
# Set before importing NumPy; no external thread-control library is required.
for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name] = '2'
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PARAMS = ['W1','b1','W2','b2']

def read_protocol(root=ROOT):
    return json.loads((root/'protocol.json').read_text(encoding='utf-8'))

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def split_indices(y,cfg):
    rng=np.random.default_rng(cfg['split']['seed']);tr=[];va=[]
    k=cfg['split']['validation_per_class']
    for digit in range(10):
        ids=rng.permutation(np.flatnonzero(y==digit));va.extend(ids[:k]);tr.extend(ids[k:])
    return np.sort(np.array(tr,dtype=np.int64)),np.sort(np.array(va,dtype=np.int64))

def load_training(root,cfg):
    with np.load(root/'data/mnist.npz',allow_pickle=False) as f:
        # Deliberately do not read official test arrays during training/benchmark.
        raw=f['x_train'];y=f['y_train'].astype(np.int64)
    assert raw.shape==(60000,28,28) and y.shape==(60000,)
    assert raw.dtype==np.uint8 and np.array_equal(np.unique(y),np.arange(10))
    tr,va=split_indices(y,cfg)
    x=(raw.reshape(-1,784).astype(np.float32)/cfg['dataset']['scale_divisor'])
    return x,y,tr,va

def load_testing(root,cfg):
    with np.load(root/'data/mnist.npz',allow_pickle=False) as f:
        raw=f['x_test'];y=f['y_test'].astype(np.int64)
    assert raw.shape==(10000,28,28) and y.shape==(10000,)
    return raw.reshape(-1,784).astype(np.float32)/cfg['dataset']['scale_divisor'],y

def shift_images(x,dy,dx):
    a=x.reshape(-1,28,28);b=np.zeros_like(a)
    ys,ye=max(0,-dy),min(28,28-dy);xs,xe=max(0,-dx),min(28,28-dx)
    b[:,ys+dy:ye+dy,xs+dx:xe+dx]=a[:,ys:ye,xs:xe]
    return b.reshape(-1,784)

def initialize(seed,cfg):
    ni,nh,no=cfg['model']['layers'];rng=np.random.default_rng(seed)
    return {'W1':(rng.normal(size=(ni,nh))*np.sqrt(2/ni)).astype(np.float32),
            'b1':np.zeros(nh,np.float32),
            'W2':(rng.normal(size=(nh,no))*np.sqrt(1/nh)).astype(np.float32),
            'b2':np.zeros(no,np.float32)}

def augment(x,method,shift_rng,noise_rng,cfg):
    c=cfg['augmentation'];mask=shift_rng.random(len(x))<c['shift_probability']
    choices=shift_rng.integers(0,4,len(x))
    nm=noise_rng.random(len(x))<c['noise_probability']
    noise=noise_rng.normal(0,c['noise_std'],x.shape).astype(np.float32)
    out=x.copy()
    if method in ['shift','joint']:
        for i,(dy,dx) in enumerate(c['direction_dy_dx']):
            use=mask&(choices==i)
            out[use]=shift_images(out[use],dy*c['shift_pixels'],dx*c['shift_pixels'])
    if method in ['noise','joint']:
        out[nm]=np.clip(out[nm]+noise[nm],0,1)
    return out

def forward(p,x):
    h=np.maximum(x@p['W1']+p['b1'],0)
    z=h@p['W2']+p['b2'];z-=z.max(axis=1,keepdims=True)
    prob=np.exp(z);prob/=prob.sum(axis=1,keepdims=True)
    return h,prob

def predict(p,x):
    return (np.maximum(x@p['W1']+p['b1'],0)@p['W2']+p['b2']).argmax(axis=1)

def loss_grad(p,x,y,l2):
    h,prob=forward(p,x)
    loss=-np.log(np.maximum(prob[np.arange(len(y)),y],1e-12)).mean()
    loss+=l2*.5*(np.sum(p['W1']**2)+np.sum(p['W2']**2))
    dz=prob.copy();dz[np.arange(len(y)),y]-=1;dz/=len(y)
    dh=(dz@p['W2'].T)*(h>0)
    return float(loss),{'W1':x.T@dh+l2*p['W1'],'b1':dh.sum(axis=0),
                        'W2':h.T@dz+l2*p['W2'],'b2':dz.sum(axis=0)}

def condition_batches(x,y,condition,phase,cfg,batch_size=None):
    """Yield deterministic views, direction-major and then image-major."""
    bs=batch_size or cfg['evaluation_batch_size'];c=cfg[phase]
    dirs=cfg['augmentation']['direction_dy_dx']
    if condition.startswith('joint_'):
        moves=[(dy*cfg['augmentation']['shift_pixels'],dx*cfg['augmentation']['shift_pixels']) for dy,dx in dirs]
        std=float(condition.split('_')[1]);rng=np.random.default_rng(c['joint_noise_seed'])
    elif condition.startswith('shift'):
        pixels=int(condition[5:]);moves=[(dy*pixels,dx*pixels) for dy,dx in dirs];std=None;rng=None
    elif condition.startswith('noise_'):
        moves=[(0,0)];std=float(condition.split('_')[1]);rng=np.random.default_rng(c['noise_seed'])
    elif condition=='clean':moves=[(0,0)];std=None;rng=None
    else:raise ValueError(condition)
    for dy,dx in moves:
        for start in range(0,len(y),bs):
            z=x[start:start+bs]
            if dy or dx:z=shift_images(z,dy,dx)
            if std is not None:
                eps=rng.normal(size=z.shape).astype(np.float32)
                z=np.clip(z+std*eps,0,1)
            yield z,y[start:start+bs]

def materialize_validation(x,y,cfg):
    cached={};hashes={}
    for condition in cfg['validation']['conditions']:
        xs=[];ys=[];hx=hashlib.sha256();hy=hashlib.sha256()
        for bx,by in condition_batches(x,y,condition,'validation',cfg):
            xs.append(bx);ys.append(by);hx.update(bx.astype('<f4',copy=False).tobytes());hy.update(by.astype('<i8',copy=False).tobytes())
        cached[condition]=(np.concatenate(xs),np.concatenate(ys))
        hashes[condition]={'images_sha256':hx.hexdigest(),'labels_sha256':hy.hexdigest(),'n_views':sum(map(len,ys))}
    return cached,hashes

def validation_scores(p,cached,cfg):
    scores={};bs=cfg['evaluation_batch_size']
    for name,(x,y) in cached.items():
        correct=0
        for start in range(0,len(y),bs):correct+=int(np.sum(predict(p,x[start:start+bs])==y[start:start+bs]))
        scores[name]=correct/len(y)
    return scores

def train_epoch(p,x,y,idx,method,state,cfg):
    order=state['shuffle'].permutation(len(idx));total_loss=0.;opt=cfg['optimizer'];bs=cfg['batch_size']
    for start in range(0,len(idx),bs):
        ids=idx[order[start:start+bs]]
        bx=augment(x[ids],method,state['shift'],state['noise'],cfg)
        loss,grads=loss_grad(p,bx,y[ids],opt['l2_coefficient'])
        if not np.isfinite(loss):raise FloatingPointError('Non-finite training loss')
        total_loss+=loss*len(ids);state['step']+=1;t=state['step']
        for name in PARAMS:
            state['m'][name]=opt['beta1']*state['m'][name]+(1-opt['beta1'])*grads[name]
            state['v'][name]=opt['beta2']*state['v'][name]+(1-opt['beta2'])*grads[name]**2
            p[name]-=opt['learning_rate']*(state['m'][name]/(1-opt['beta1']**t))/(np.sqrt(state['v'][name]/(1-opt['beta2']**t))+opt['epsilon'])
    return total_loss/len(idx)

def training_state(p,seed):
    return {'m':{k:np.zeros_like(v) for k,v in p.items()},'v':{k:np.zeros_like(v) for k,v in p.items()},
            'shuffle':np.random.default_rng(seed+1000),'shift':np.random.default_rng(seed+2000),
            'noise':np.random.default_rng(seed+3000),'step':0}

def metrics_from_confusion(cm):
    diag=np.diag(cm);den=cm.sum(axis=0)+cm.sum(axis=1)
    f1=np.divide(2*diag,den,out=np.zeros(10,dtype=float),where=den!=0)
    return float(diag.sum()/cm.sum()),float(f1.mean())

def evaluate_condition(p,x,y,condition,cfg):
    preds=[];cm=np.zeros((10,10),dtype=np.int64);hx=hashlib.sha256();hy=hashlib.sha256()
    for bx,by in condition_batches(x,y,condition,'test',cfg):
        pred=predict(p,bx);preds.append(pred.astype(np.uint8))
        cm+=np.bincount(by*10+pred,minlength=100).reshape(10,10)
        hx.update(bx.astype('<f4',copy=False).tobytes());hy.update(by.astype('<i8',copy=False).tobytes())
    acc,f1=metrics_from_confusion(cm)
    return np.concatenate(preds),cm,{'accuracy':acc,'macro_f1':f1,'correct':int(np.trace(cm)),'n_views':int(cm.sum())}, {'images_sha256':hx.hexdigest(),'labels_sha256':hy.hexdigest(),'n_views':int(cm.sum())}

def benchmark(root):
    cfg=read_protocol(root);x,y,tr,va=load_training(root,cfg)
    t=time.perf_counter();cached,_=materialize_validation(x[va],y[va],cfg);prep=time.perf_counter()-t
    p=initialize(cfg['seeds'][0],cfg);state=training_state(p,cfg['seeds'][0])
    t=time.perf_counter();loss=train_epoch(p,x,y,tr,'baseline',state,cfg);train_s=time.perf_counter()-t
    t=time.perf_counter();scores=validation_scores(p,cached,cfg);val_s=time.perf_counter()-t
    print(json.dumps({'benchmark':'one baseline epoch, official test arrays untouched','train_seconds':train_s,
                      'validation_seconds':val_s,'validation_cache_seconds':prep,'train_loss':loss,
                      'validation_score':float(np.mean(list(scores.values()))),
                      'estimated_200_epoch_seconds_excluding_test':200*(train_s+val_s)},indent=2),flush=True)

def run(root):
    started=time.perf_counter();cfg=read_protocol(root)
    assert cfg['epochs']==10 and cfg['blas_threads']==2
    for sub in ['weights','results/logs','results/predictions','results/confusions','figures']:(root/sub).mkdir(parents=True,exist_ok=True)
    x,y,tr,va=load_training(root,cfg)
    assert len(tr)==55000 and len(va)==5000 and not set(tr)&set(va)
    np.savez_compressed(root/'data/split_indices.npz',train_idx=tr,validation_idx=va)
    cache_t=time.perf_counter();cached,val_hashes=materialize_validation(x[va],y[va],cfg)
    meta={'started_at_local':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'python':sys.version,'numpy':np.__version__,
          'platform':platform.platform(),'processor':platform.processor(),'thread_environment':{k:os.environ[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
          'dataset_sha256':sha256(root/'data/mnist.npz'),'protocol_sha256':sha256(root/'protocol.json'),
          'training_script_sha256_at_run':sha256(Path(__file__)),'split_counts':{'training':len(tr),'validation':len(va),'official_test':10000},
          'training_class_counts':np.bincount(y[tr],minlength=10).tolist(),'validation_class_counts':np.bincount(y[va],minlength=10).tolist(),
          'parameter_count':sum(a.size for a in initialize(cfg['seeds'][0],cfg).values()),
          'validation_cache_seconds':time.perf_counter()-cache_t,'validation_input_hashes':val_hashes,'runs':[]}
    write_json(root/'results/environment.json',meta)
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            run_t=time.perf_counter();p=initialize(seed,cfg);state=training_state(p,seed);best=-1.;best_epoch=0;logs=[]
            for epoch in range(1,cfg['epochs']+1):
                ep_t=time.perf_counter();loss=train_epoch(p,x,y,tr,method,state,cfg)
                vals=validation_scores(p,cached,cfg);score=float(np.mean(list(vals.values())))
                logs.append({'epoch':epoch,'train_loss':loss,'validation_score':score,
                             **{'val_'+k:v for k,v in vals.items()},'epoch_seconds_including_validation':time.perf_counter()-ep_t})
                if score>best:
                    best=score;best_epoch=epoch
                    np.savez_compressed(root/f'weights/{method}_seed{seed}.npz',**p,epoch=np.array(epoch),validation_score=np.array(score),seed=np.array(seed),method=np.array(method))
                print(json.dumps({'stage':'training','seed':seed,'method':method,'epoch':epoch,'validation_score':score,'seconds':logs[-1]['epoch_seconds_including_validation']}),flush=True)
            write_csv(root/f'results/logs/{method}_seed{seed}.csv',logs)
            meta['runs'].append({'method':method,'seed':seed,'best_epoch':best_epoch,'best_validation_score':best,'steps':state['step'],'training_seconds_including_validation':time.perf_counter()-run_t})
            write_json(root/'results/environment.json',meta)
    meta['all_training_finished_at_local']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    write_json(root/'results/environment.json',meta)
    del cached,x,y
    tx,ty=load_testing(root,cfg);rows=[];test_hashes={}
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            with np.load(root/f'weights/{method}_seed{seed}.npz',allow_pickle=False) as f:p={k:f[k] for k in PARAMS}
            saved={};cms={}
            for condition in cfg['test']['conditions']:
                pred,cm,metric,hashes=evaluate_condition(p,tx,ty,condition,cfg)
                if condition in test_hashes:assert test_hashes[condition]==hashes
                else:test_hashes[condition]=hashes
                saved[condition]=pred;cms[condition]=cm
                rows.append({'method':method,'seed':seed,'condition':condition,**metric})
            np.savez_compressed(root/f'results/predictions/{method}_seed{seed}.npz',**saved)
            np.savez_compressed(root/f'results/confusions/{method}_seed{seed}.npz',**cms)
            print(json.dumps({'stage':'test','seed':seed,'method':method,'conditions_evaluated':len(saved)}),flush=True)
    write_csv(root/'results/metrics.csv',rows);write_json(root/'results/metrics.json',rows)
    summary=[]
    for method in cfg['methods']:
        for condition in cfg['test']['conditions']:
            rr=[r for r in rows if r['method']==method and r['condition']==condition]
            summary.append({'method':method,'condition':condition,'n_seeds':len(rr),
                            **{key+'_'+stat:float(getattr(np,stat)([r[key] for r in rr],**({'ddof':1} if stat=='std' else {}))) for key in ['accuracy','macro_f1'] for stat in ['mean','std']}})
    write_csv(root/'results/summary.csv',summary);write_json(root/'results/summary.json',summary)
    write_json(root/'results/test_input_hashes.json',test_hashes)
    meta['official_test_class_counts']=np.bincount(ty,minlength=10).tolist()
    meta['completed_at_local']=time.strftime('%Y-%m-%dT%H:%M:%S%z');meta['total_wall_seconds']=time.perf_counter()-started
    write_json(root/'results/environment.json',meta)
    # Only immutable core experiment records enter this manifest; derived plots/analyses are excluded.
    fixed=[root/'data/mnist.npz',root/'data/split_indices.npz',root/'protocol.json',root/'train.py']
    for folder in ['weights','results/logs','results/predictions','results/confusions']:fixed.extend((root/folder).glob('*.npz' if folder!='results/logs' else '*.csv'))
    fixed.extend(root/'results'/n for n in ['metrics.csv','metrics.json','summary.csv','summary.json','environment.json','test_input_hashes.json'])
    (root/'SHA256SUMS.txt').write_text('\n'.join(sha256(f)+'  '+f.relative_to(root).as_posix() for f in sorted(fixed))+'\n',encoding='utf-8')
    print(json.dumps({'stage':'complete','total_seconds':meta['total_wall_seconds']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    if args.benchmark:benchmark(ROOT)
    else:run(ROOT)
