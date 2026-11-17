import numpy as np
from scipy.stats import binom
from sklearn.neighbors import KNeighborsClassifier
def h(p):
    p=np.clip(p,1e-12,1-1e-12); return -(p*np.log2(p)+(1-p)*np.log2(1-p))
eta=lambda x: 0.5+0.4*np.sin(2*np.pi*x)
rng=np.random.default_rng(0); xt=rng.random(400)[:,None]; et=eta(xt[:,0])
res={}
for k in [1,3,9]:
    beta=1-binom.cdf(k//2,k,et); lim=h(np.minimum(beta,1-beta)).mean()
    exlim=np.mean(np.abs(2*et-1)*np.minimum(beta,1-beta))
    for n in [200,2000,20000]:
        P=[]
        for r in range(300):
            x=rng.random(n)[:,None]; y=(rng.random(n)<eta(x[:,0])).astype(int)
            P.append(KNeighborsClassifier(k).fit(x,y).predict(xt))
        p1=np.mean(P,0); pi=np.minimum(p1,1-p1); V=np.where(pi>0,h(pi),0).mean()
        bstar=(et>0.5).astype(int); ex=np.mean(np.abs(2*et-1)*np.mean(np.array(P)!=bstar,0))
        print(f'k={k} n={n}: V={V:.3f} (limit {lim:.3f})  excess={ex:.4f} (limit {exlim:.4f})',flush=True)
print('H(Y|X)=%.3f'%h(et).mean())
