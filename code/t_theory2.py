import numpy as np, warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
def h(p):
    p=np.clip(p,1e-12,1-1e-12); return -(p*np.log2(p)+(1-p)*np.log2(1-p))
rng=np.random.default_rng(0)
eta=lambda x: 0.5+0.4*np.sin(2*np.pi*x)
# (a) k_n-NN rate: k = n^{2/3}, d = 1, margin exponent alpha = 1 -> predicted E V = O(n^{-1/3})
xt=rng.random(1000)[:,None]; et=eta(xt[:,0]); b=(et>0.5).astype(int); w=np.abs(2*et-1)
res=[]
for n in [250,1000,4000,16000]:
    k=int(round(n**(2/3))) | 1; P=[]
    for r in range(150):
        x=rng.random(n)[:,None]; y=(rng.random(n)<eta(x[:,0])).astype(int); P.append(KNeighborsClassifier(k).fit(x,y).predict(xt))
    P=np.array(P); p=P.mean(0); pi=np.minimum(p,1-p); V=np.where(pi>0,h(pi),0).mean(); ex=np.mean(w*np.mean(P!=b,0))
    res.append((n,k,V,ex)); print(f"n={n} k={k} EV={V:.4f} excess={ex:.5f}",flush=True)
n=np.log([r[0] for r in res]); print("slope log EV vs log n: %.3f (predicted <= -1/3)"%np.polyfit(n,np.log([r[2] for r in res]),1)[0],
      " slope excess: %.3f"%np.polyfit(n,np.log([r[3] for r in res]),1)[0])
# (b) Heskes decomposition for log-loss with logistic regression on a misspecified eta
X2=rng.random(400)[:,None]; e2=eta(X2[:,0]); Q=[]
for r in range(300):
    x=rng.random(150)[:,None]; y=(rng.random(150)<eta(x[:,0])).astype(int)
    Q.append(LogisticRegression(C=10).fit(x,y).predict_proba(X2)[:,1])
Q=np.clip(np.array(Q),1e-9,1-1e-9); E=np.stack([1-e2,e2],1)
KL=lambda a,b_: (a*np.log(a/b_)).sum(-1)
lhs=np.mean([KL(E,np.stack([1-q,q],1)) for q in Q],0)
G=np.exp(np.stack([np.log(1-Q).mean(0),np.log(Q).mean(0)],1)); Z=G.sum(1); qbar=G/Z[:,None]
rhs=KL(E,qbar)+(-np.log(Z))
print("Heskes: max |E KL(eta||q_S) - [KL(eta||qbar) + (-log Z)]| = %.2e ; mean variance term -log Z = %.4f nats"%(np.abs(lhs-rhs).max(),(-np.log(Z)).mean()))
# (c) ridgeless: prediction variance over label noise equals sigma^2 ||(X^+)^T x'||^2
nn,p=50,200; X=rng.normal(size=(nn,p)); xp=rng.normal(size=(20,p)); beta=rng.normal(size=p)/np.sqrt(p); s=0.5
Xp=np.linalg.pinv(X); F=np.array([xp@(Xp@(X@beta+s*rng.normal(size=nn))) for _ in range(20000)])
print("ridgeless: max rel. err of MC variance vs formula = %.3f"%np.max(np.abs(F.var(0)/(s**2*np.sum((xp@Xp)**2,1))-1)))
