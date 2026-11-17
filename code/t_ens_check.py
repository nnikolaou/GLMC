import numpy as np
from scipy.stats import binom
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
def h(p):
    p=np.clip(p,1e-12,1-1e-12); return -(p*np.log2(p)+(1-p)*np.log2(1-p))
import sys
F=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
eta=lambda x: 0.5+0.4*np.sin(2*np.pi*F*x)
rng=np.random.default_rng(1); xt=rng.random(400)[:,None]; et=eta(xt[:,0]); bstar=(et>0.5).astype(int); mg=np.abs(2*et-1)
for name,mk in [('1-NN',lambda:KNeighborsClassifier(1)),('tree depth 2',lambda:DecisionTreeClassifier(max_depth=2)),('logistic (misspecified)',lambda:LogisticRegression())]:
    P=[]
    for r in range(25*40):
        x=rng.random(200)[:,None]; y=(rng.random(200)<eta(x[:,0])).astype(int); P.append(mk().fit(x,y).predict(xt))
    P=np.array(P); p=P.mean(0); pi=np.minimum(p,1-p); m=(p>=0.5).astype(int)
    for B in [1,5,25]:
        G=(P[:(len(P)//B)*B].reshape(-1,B,len(xt)).mean(1)>0.5).astype(int)
        sim=np.mean(mg*np.mean(G!=bstar,0)); piB=1-binom.cdf(B//2,B,pi)
        pred=np.mean(mg*np.where(m==bstar,piB,1-piB)); VB=h(np.minimum(G.mean(0),1-G.mean(0))).mean()
        print(f'{name} B={B}: simulated excess {sim:.4f}, predicted {pred:.4f}, V_B {VB:.3f}')
    lim=np.mean(mg*pi*((m!=bstar).astype(float)-(m==bstar))*(pi<0.5)); print(f'  predicted B->inf change in excess: {lim:+.4f}; modal-bias mass P(m!=b*)={np.mean(m!=bstar):.3f}')
