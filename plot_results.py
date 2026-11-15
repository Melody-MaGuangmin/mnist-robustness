"""Generate seven figures and auditable sample records from the MNIST run only.

Requires matplotlib in addition to the training requirements. This script does not
train, select checkpoints, modify data, or overwrite the core experiment records.
"""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import ROOT,PARAMS,read_protocol,load_testing,condition_batches,predict,sha256,write_json

METHODS=['baseline','shift','noise','joint']
SEEDS=[11,22,33,44,55]
NAMES={'baseline':'Baseline','shift':'Shift only','noise':'Noise only','joint':'Joint'}
COLORS={'baseline':'#444444','shift':'#2371A5','noise':'#C58B23','joint':'#A23B45'}
DIR_NAMES=['up','down','left','right']
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':220})

def save(fig,name):
    path=ROOT/'figures'/name
    fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig)
    print(str(path),flush=True)

def selected_views(x,y,condition,indices,cfg):
    wanted=set(map(int,indices));out={};offset=0
    for bx,by in condition_batches(x,y,condition,'test',cfg):
        end=offset+len(by)
        for index in sorted(wanted):
            if offset<=index<end:out[index]=bx[index-offset].copy()
        offset=end
        if len(out)==len(wanted):break
    assert len(out)==len(wanted)
    return np.stack([out[int(i)] for i in indices])

def summary_plots(s,cfg):
    conds=cfg['test']['conditions']
    labels=['Clean','Noise\n0.10','Noise\n0.20','Noise\n0.30','Shift\n2 px','Shift 2 px\n+ noise 0.20','Shift\n4 px']
    fig,ax=plt.subplots(figsize=(9.8,4.5));positions=np.arange(len(conds));width=.19
    for mi,m in enumerate(METHODS):
        vals=[100*s[m,c]['accuracy_mean'] for c in conds]
        err=[100*s[m,c]['accuracy_std'] for c in conds]
        ax.bar(positions+(mi-1.5)*width,vals,width,color=COLORS[m],label=NAMES[m],yerr=err,capsize=2,error_kw={'lw':.8})
    ax.set(xticks=positions,xticklabels=labels,ylabel='Accuracy (%)',ylim=(0,103));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.14),frameon=False);fig.tight_layout();save(fig,'02_accuracy.png')
    fig,ax=plt.subplots(figsize=(7,3.8))
    for m in METHODS:
        cs=['clean','noise_0.10','noise_0.20','noise_0.30']
        ax.errorbar([0,.1,.2,.3],[100*s[m,c]['accuracy_mean'] for c in cs],
                    yerr=[100*s[m,c]['accuracy_std'] for c in cs],color=COLORS[m],label=NAMES[m],marker='o',capsize=3)
    ax.set(xlabel='Gaussian noise standard deviation',ylabel='Accuracy (%)',xticks=[0,.1,.2,.3]);ax.legend(ncol=2,frameon=False);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'03_noise_curve.png')
    fig,ax=plt.subplots(figsize=(7,3.8))
    for m in METHODS:
        records=[]
        for seed in SEEDS:
            with open(ROOT/f'results/logs/{m}_seed{seed}.csv',encoding='utf-8',newline='') as f:
                records.append([float(r['validation_score']) for r in csv.DictReader(f)])
        a=np.array(records)*100;mean=a.mean(0);sd=a.std(0,ddof=1);ep=np.arange(1,a.shape[1]+1)
        ax.plot(ep,mean,color=COLORS[m],label=NAMES[m],marker='o',markersize=3)
        ax.fill_between(ep,mean-sd,mean+sd,color=COLORS[m],alpha=.12)
    ax.set(xlabel='Epoch',ylabel='Mixed validation accuracy (%)',xticks=np.arange(1,cfg['epochs']+1));ax.legend(ncol=2,frameon=False);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'04_validation_curve.png')

def confusion_plot():
    fig,axs=plt.subplots(1,2,figsize=(9.2,4),constrained_layout=True);details={}
    for m in METHODS:
        cms=[]
        for seed in SEEDS:
            with np.load(ROOT/f'results/confusions/{m}_seed{seed}.npz',allow_pickle=False) as f:cms.append(f['joint_0.20'])
        mean=np.mean(cms,axis=0);norm=100*mean/mean.sum(axis=1,keepdims=True)
        details[m]={'recall_pct':np.diag(norm).tolist(),'mean_confusion_counts':mean.tolist(),'row_normalized_pct':norm.tolist()}
        if m not in ['baseline','joint']:continue
        ax=axs[0 if m=='baseline' else 1];im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=100)
        for i in range(10):
            for j in range(10):
                if i==j or norm[i,j]>=5:ax.text(j,i,f'{norm[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if norm[i,j]>55 else 'black')
        ax.set(title=NAMES[m],xlabel='Predicted digit',ylabel='True digit',xticks=range(10),yticks=range(10))
    fig.colorbar(im,ax=axs,label='Row-normalized frequency (%)',shrink=.85);save(fig,'05_confusion.png')
    return details

def image_plots(tx,ty,cfg):
    n=len(ty);ids=[int(np.flatnonzero(ty==digit)[0]) for digit in range(10)]
    clean=tx[ids];shifted=selected_views(tx,ty,'shift2',ids,cfg)
    noisy=selected_views(tx,ty,'noise_0.20',ids,cfg);joint=selected_views(tx,ty,'joint_0.20',ids,cfg)
    fig,axs=plt.subplots(4,10,figsize=(10.2,4.1),gridspec_kw={'wspace':.035,'hspace':.1})
    for ri,(row,label) in enumerate([(clean,'Clean'),(shifted,'Shift up 2 px'),(noisy,'Noise 0.20'),(joint,'Shift + noise')]):
        for ci,img in enumerate(row):
            ax=axs[ri,ci];ax.imshow(img.reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if ri==0:ax.set_title(str(ci))
            if ci==0:ax.set_ylabel(label,rotation=0,ha='right',va='center')
    save(fig,'01_examples.png')
    preds={};params={};weight_hashes={}
    for m in ['baseline','joint']:
        with np.load(ROOT/f'results/predictions/{m}_seed11.npz',allow_pickle=False) as f:
            preds[m]={c:f[c] for c in ['clean','joint_0.20']}
        wp=ROOT/f'weights/{m}_seed11.npz';weight_hashes[m]=sha256(wp)
        with np.load(wp,allow_pickle=False) as f:params[m]={k:f[k] for k in PARAMS}
        assert np.array_equal(predict(params[m],clean),preds[m]['clean'][ids])
    records=[]
    fig,axs=plt.subplots(2,5,figsize=(9.4,5.0),gridspec_kw={'hspace':.58,'wspace':.12})
    for ci,(idx,ax) in enumerate(zip(ids,axs.ravel())):
        b=int(preds['baseline']['clean'][idx]);j=int(preds['joint']['clean'][idx]);truth=int(ty[idx])
        ax.imshow(tx[idx].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
        ax.set_title(f'True digit: {truth}',fontsize=10)
        ax.text(.5,-.07,f'Baseline: {b}   Joint: {j}',transform=ax.transAxes,ha='center',va='top',fontsize=9)
        ax.text(.5,-.21,f'Official test index: {idx}',transform=ax.transAxes,ha='center',va='top',fontsize=8,color='#555555')
        records.append({'figure':'clean','condition':'clean','view_index':idx,'official_test_index':idx,'true':truth,'baseline_prediction':b,'joint_prediction':j})
    save(fig,'06_clean_predictions.png')
    truth_all=np.tile(ty,4);bp=preds['baseline']['joint_0.20'];jp=preds['joint']['joint_0.20']
    groups=[('Both correct',(bp==truth_all)&(jp==truth_all)),
            ('Baseline wrong / joint correct',(bp!=truth_all)&(jp==truth_all)),
            ('Baseline correct / joint wrong',(bp==truth_all)&(jp!=truth_all)),
            ('Both wrong',(bp!=truth_all)&(jp!=truth_all))]
    chosen=[];used=set();group_counts={}
    for name,mask in groups:
        found=[];group_counts[name]=int(np.sum(mask))
        for idx in np.flatnonzero(mask):
            original=int(idx)%n
            if original in used:continue
            found.append(int(idx));used.add(original)
            if len(found)==3:break
        if len(found)!=3:raise RuntimeError(f'Not enough distinct examples for {name}')
        chosen.append(found)
    flat=[i for group in chosen for i in group];mixed=selected_views(tx,ty,'joint_0.20',flat,cfg)
    for m in ['baseline','joint']:assert np.array_equal(predict(params[m],mixed),preds[m]['joint_0.20'][flat])
    fig=plt.figure(figsize=(11.4,10.2));gs=fig.add_gridspec(4,6,wspace=.12,hspace=.70,left=.05,right=.98,top=.94,bottom=.06)
    for gi,((name,_),indices) in enumerate(zip(groups,chosen)):
        for ci,idx in enumerate(indices):
            pos=idx%n;direction=DIR_NAMES[idx//n];truth=int(ty[pos]);b=int(bp[idx]);j=int(jp[idx])
            orig=fig.add_subplot(gs[gi,2*ci]);corr=fig.add_subplot(gs[gi,2*ci+1])
            for ax,img in [(orig,tx[pos]),(corr,mixed[gi*3+ci])]:
                ax.imshow(img.reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            orig.set_title('Original',fontsize=9);corr.set_title('Actual test input',fontsize=9)
            orig.set_xlabel(f'Index {pos}',fontsize=8)
            corr.set_xlabel(f'True {truth} | B {b} | J {j}\n{direction} 2 px + noise 0.20',fontsize=8)
            if ci==0:
                orig.text(0,1.34,name,transform=orig.transAxes,ha='left',va='bottom',fontsize=11,fontweight='bold')
            records.append({'figure':'mixed','group':name,'condition':'joint_0.20','view_index':idx,
                            'official_test_index':pos,'direction':direction,'true':truth,
                            'baseline_prediction':b,'joint_prediction':j})
    fig.suptitle('Seed 11: balanced examples of agreement and disagreement',fontsize=12,y=.995)
    save(fig,'07_mixed_cases.png')
    selection={'seed':11,'selection_rule':'clean: first official test image per class; mixed: first 3 in stored view order per category, without repeating original images across the four categories',
               'note':'The categories are deliberately balanced for explanation, not sampled according to their population frequencies.',
               'joint_condition_view_count':4*n,'category_counts_over_all_views':group_counts,
               'weights_sha256':weight_hashes,'selected_predictions_reloaded_and_matched':True,'records':records}
    write_json(ROOT/'results/visualization_selection.json',selection)
    return selection

def main():
    cfg=read_protocol();(ROOT/'figures').mkdir(exist_ok=True)
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    raw=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))
    assert len(summary)==28 and len(raw)==140
    s={(r['method'],r['condition']):r for r in summary}
    r={(r['method'],r['seed'],r['condition']):r for r in raw}
    summary_plots(s,cfg);per_class=confusion_plot();tx,ty=load_testing(ROOT,cfg);selection=image_plots(tx,ty,cfg)
    extended={'paired_changes':{},'mixed_per_class':per_class,'seed11_metrics':{},'visualization_selection_file':'results/visualization_selection.json'}
    for other in ['baseline','shift','noise']:
        key='joint_minus_'+other;extended['paired_changes'][key]={}
        for c in cfg['test']['conditions']:
            diffs=[100*(r['joint',seed,c]['accuracy']-r[other,seed,c]['accuracy']) for seed in SEEDS]
            b=s[other,c]['accuracy_mean'];j=s['joint',c]['accuracy_mean']
            extended['paired_changes'][key][c]={'seeds':SEEDS,'values_pp':diffs,'mean_pp':float(np.mean(diffs)),
                'sample_sd_pp':float(np.std(diffs,ddof=1)),'min_pp':min(diffs),'max_pp':max(diffs),
                'relative_error_reduction':float((j-b)/(1-b)) if b<1 else None}
    for m in METHODS:
        extended['seed11_metrics'][m]={c:{k:r[m,11,c][k] for k in ['accuracy','macro_f1','correct','n_views']} for c in cfg['test']['conditions']}
    environment=json.loads((ROOT/'results/environment.json').read_text(encoding='utf-8'))
    extended['best_epochs']={m:[q['best_epoch'] for q in environment['runs'] if q['method']==m] for m in METHODS}
    extended['runtime_seconds']=environment['total_wall_seconds']
    extended['mixed_case_category_counts']=selection['category_counts_over_all_views']
    write_json(ROOT/'results/extended_analysis.json',extended)
    write_json(ROOT/'results/figure_environment.json',{'python':sys.version,'numpy':np.__version__,
        'matplotlib':matplotlib.__version__,'plot_script_sha256':sha256(Path(__file__)),
        'source_metrics_sha256':sha256(ROOT/'results/metrics.json'),
        'figures_sha256':{f.name:sha256(f) for f in sorted((ROOT/'figures').glob('*.png'))}})
    print('Seven MNIST figures and extended analyses generated from actual results.',flush=True)

if __name__=='__main__':main()
