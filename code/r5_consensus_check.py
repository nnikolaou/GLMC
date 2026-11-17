import numpy as np, glob, json
from r5_analyse import stats
rng=np.random.default_rng(0); out={}
def lf(mt,pibar,K,members):
    cons=np.array([np.bincount(col,minlength=K).argmax() for col in mt[members].T])
    D=(mt!=cons).mean(1); return D+pibar*np.clip(1-D*K/(K-1),0,1)
res={k:{'sym':[],'cd':[]} for k in ['all19','rand10','rand5','interp_only','noninterp_only','drop_best3']}
for f in sorted(glob.glob('r5_*.npz')):
    z=np.load(f); kind=f.split('_')[2]; g='cd' if kind=='classdep' else 'sym'
    Pv,Pt,yt,K=z['Pv'],z['Pt'],z['yt'],int(z['K']); test=(Pt!=yt).mean((1,2)); L=len(test)
    V,pibar,mt,mv=stats(Pv,Pt,K); interp=z['train_err']<0.02
    reg=lambda c: test[np.argmin(c)]-test.min()
    res['all19'][g].append(reg(lf(mt,pibar,K,np.arange(L))))
    res['rand10'][g].append(np.mean([reg(lf(mt,pibar,K,rng.choice(L,10,replace=False))) for _ in range(30)]))
    res['rand5'][g].append(np.mean([reg(lf(mt,pibar,K,rng.choice(L,5,replace=False))) for _ in range(30)]))
    if interp.sum()>=3: res['interp_only'][g].append(reg(lf(mt,pibar,K,np.flatnonzero(interp))))
    if (~interp).sum()>=3: res['noninterp_only'][g].append(reg(lf(mt,pibar,K,np.flatnonzero(~interp))))
    res['drop_best3'][g].append(reg(lf(mt,pibar,K,np.argsort(test)[3:])))
for k,v in res.items(): print(f"{k:15s} sym {np.mean(v['sym']):.4f} (n={len(v['sym'])})  classdep {np.mean(v['cd']):.4f} (n={len(v['cd'])})")
