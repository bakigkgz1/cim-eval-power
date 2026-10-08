"""Sequential (early-stopping) evaluation audit, Table 10 of the manuscript.
Usage: python sequential_audit.py [path-to-repo-root]
Runs group-sequential paired non-inferiority tests (Pocock and O'Brien-Fleming boundaries,
looks at 250..10000 images) and an empirical-Bernstein confidence sequence on 1000 random
orderings of each checkpoint's full test set and compares the verdicts with the full-test verdict.
"""
import numpy as np, glob, json, os, csv
import sys
REPO=sys.argv[1] if len(sys.argv)>1 else '.'
R=f'{REPO}/data/results/fulltest10k'
ck = {
 'VGG8/C10 (a)':'c10/logits_vgg8_p3', 'VGG8/C10 (b)':'c10/logits_vgg8_2nd',
 'R18/C10 (a)':'c10/logits_resnet_c10_p3','R18/C10 s5678':'c10/logits_resnet_c10_5678',
 'R18/C100 s34':'c100/logits_s34','R18/C100 s55':'c100/logits_s55','R18/C100 s13':'c100/logits_s13',
 'R18/C100 s21':'c100/logits_s21','R18/C100 s89':'c100/logits_s89','R18/C100 s5678':'c100/logits_P4',
 'VGG8/C100 s13':'c100_vgg8/logits_vgg8c100_s13','VGG8/C100 s21':'c100_vgg8/logits_vgg8c100_s21',
 'VGG8/C100 s34':'c100_vgg8/logits_vgg8c100_s34','VGG8/C100 s55':'c100_vgg8/logits_vgg8c100_s55',
 'VGG8/C100 s89':'c100_vgg8/logits_vgg8c100_s89'}
# accuracy-matched CIFAR-10 arm (G16), processed as a second block with its own generator
ck2 = {f'VGG8/C10m s{s}': f'c10m_vgg8/logits_vgg8c10low_s{s}' for s in (13, 21, 34, 55, 89)}
TAU=1.0; ALPHA=0.05; NPERM=1000; LOOKS=np.array([250,500,1000,2000,4000,7000,10000])

def load(p):
    r=list(csv.DictReader(open(f'{R}/{p}_n10000_per_image.csv')))
    a=np.array([int(x['correct_fp']) for x in r]); b=np.array([int(x['correct_clip']) for x in r])
    return (b-a).astype(np.int8)

# --- O'Brien-Fleming constant for the look schedule (two one-sided tests at alpha/2 each)
def obf_C(looks, alpha=ALPHA, sims=400000, seed=7):
    rng=np.random.default_rng(seed); t=looks/looks[-1]
    inc=rng.standard_normal((sims,len(t)))*np.sqrt(np.diff(np.r_[0,t]))
    W=np.cumsum(inc,1); Z=W/np.sqrt(t)
    lo,hi=1.5,4.0
    for _ in range(50):
        C=(lo+hi)/2; c=C*np.sqrt(1/t)
        p=np.mean((Z>c).any(1))   # one side
        lo,hi=(C,hi) if p>alpha/2 else (lo,C)
    return C
C=obf_C(LOOKS); CB=C*np.sqrt(LOOKS[-1]/LOOKS)
def pocock_c(looks, alpha=ALPHA, sims=400000, seed=7):
    rng=np.random.default_rng(seed); t=looks/looks[-1]
    W=np.cumsum(rng.standard_normal((sims,len(t)))*np.sqrt(np.diff(np.r_[0,t])),1); Z=W/np.sqrt(t)
    lo,hi=1.5,4.0
    for _ in range(50):
        c=(lo+hi)/2; p=np.mean((Z>c).any(1)); lo,hi=(c,hi) if p>alpha/2 else (lo,c)
    return c
CP=np.full(len(LOOKS),pocock_c(LOOKS)); print('Pocock c',round(CP[0],3))
print('OBF C',round(C,3),'boundaries',np.round(CB,3))

def gs_run(D, CB=CB):
    """group-sequential OBF on one stream D (length 10000)"""
    for k,n in enumerate(LOOKS):
        x=D[:n]; m=x.mean(); d=max(np.mean(x!=0),1/n); se=np.sqrt(max(d-m*m,1/n**2)/n)
        L,U=100*(m-CB[k]*se),100*(m+CB[k]*se)
        if L>=-TAU: return 'hold',n
        if U<-TAU: return 'fail',n
    return 'unresolved',LOOKS[-1]

def eb_cs(D, alpha=ALPHA, cmax=0.5):
    """predictable plug-in empirical-Bernstein confidence sequence (Waudby-Smith & Ramdas 2024)
    on X=(D+1)/2 in [0,1]; returns running-intersection lower/upper for Delta in pp."""
    X=(D.astype(float)+1)/2; n=X.size; t=np.arange(1,n+1)
    mu=(0.5+np.cumsum(X))/(t+1)
    mu_prev=np.r_[0.5,mu[:-1]]
    s2=(0.25+np.cumsum((X-mu)**2))/(t+1); s2_prev=np.r_[0.25,s2[:-1]]
    lam=np.minimum(np.sqrt(2*np.log(2/alpha)/(s2_prev*t*np.log(1+t))),cmax)
    v=4*(X-mu_prev)**2; psi=(-np.log(1-lam)-lam)/4
    SL=np.cumsum(lam); cen=np.cumsum(lam*X)/SL; w=(np.log(2/alpha)+np.cumsum(v*psi))/SL
    L=np.maximum.accumulate(cen-w); U=np.minimum.accumulate(cen+w)
    return 100*(2*L-1), 100*(2*U-1)

def eb_run(D):
    L,U=eb_cs(D)
    h=np.flatnonzero(L>=-TAU); f=np.flatnonzero(U<-TAU)
    ih=h[0] if h.size else 10**9; jf=f[0] if f.size else 10**9
    if ih==jf==10**9: return 'unresolved',D.size
    return ('hold',ih+1) if ih<jf else ('fail',jf+1)

full=json.load(open(f'{REPO}/verification.json'))
fv={}
for v in full:
    if not v['file'].startswith('data/results/fulltest10k/'): continue
    fv[os.path.basename(v['file']).replace('_n10000.pt','').replace('_n10000_per_image.csv','')]=v
out={}
def run_block(ckd, from_counts=False):
    rng=np.random.default_rng(1234)
    for name,p in ckd.items():
        D=load(p)
        if from_counts:
            # the order-randomized analysis depends only on the counts; a canonical order
            # makes the C10m rows of Table 10 exactly reproducible
            D=np.r_[-np.ones(int((D==-1).sum())),np.ones(int((D==1).sum())),np.zeros(int((D==0).sum()))].astype(np.int8)
        delta=100*D.mean(); truth='hold' if delta>-TAU else 'fail'
        key=os.path.basename(p); full_bin=fv[key]['verdict'] if key in fv else None
        res={'gs':[], 'pk':[], 'eb':[]}
        for r in range(NPERM):
            Dp=D[rng.permutation(D.size)]
            res['gs'].append(gs_run(Dp)); res['pk'].append(gs_run(Dp,CP)); res['eb'].append(eb_run(Dp))
        o={'delta':delta,'d':float(np.mean(D!=0)),'full_verdict':full_bin,'truth_side':truth}
        for m in ('gs','pk','eb'):
            v=np.array([x[0] for x in res[m]]); n=np.array([x[1] for x in res[m]])
            concl=v!='unresolved'
            o[m]={'hold':float(np.mean(v=='hold')),'fail':float(np.mean(v=='fail')),'unres':float(np.mean(~concl)),
                  'wrong':float(np.mean(concl&(v!=truth))),'agree_full':float(np.mean(v==full_bin)),
                  'n_med':float(np.median(n)),'n_p90':float(np.percentile(n,90,method='higher')),'n_mean':float(n.mean())}
        out[name]=o
        pr=lambda m: f"{m}: agree {o[m]['agree_full']:.2f} wrong {o[m]['wrong']:.3f} n med {o[m]['n_med']:.0f} p90 {o[m]['n_p90']:.0f}"
        print(f"{name:15s} Δ{delta:+.2f} d{o['d']:.3f} {full_bin} | "+' | '.join(pr(m) for m in ('pk','gs','eb')))
run_block(ck)
run_block(ck2, from_counts=True)
json.dump({'looks':LOOKS.tolist(),'obf_C':C,'pocock_c':float(CP[0]),'boundaries':CB.tolist(),'nperm':NPERM,'rows':out},open('sequential_audit.json','w'),indent=1)
