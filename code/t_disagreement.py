"""Disagreement vs test error vs expressed information on the running example (eps=0.2), with exact risks.
Independent copies = independent training samples (and seeds). Also the GDE setting of Jiang et al. (2022):
copies trained on the SAME sample with different seeds. Writes t_disagreement.json."""
import numpy as np, json, warnings
from pilot import sample_x, feats, eta_of_z, h2, make_learners
warnings.filterwarnings("ignore")
def run(eps=0.2, ns=(256, 1024, 4096), R=10, M=20000, seed=3):
    rng = np.random.default_rng(seed); zt, ut = sample_x(rng, M); Xt = feats(zt, ut)
    eta_t, b_t = eta_of_z(zt, eps); e1 = eta_t                      # P(Y=1|x)
    Rstar = float(np.mean(np.minimum(e1, 1 - e1))); out = []
    for n in ns:
        for name, mk in make_learners(n).items():
            P = np.zeros((R, M), np.int8); Pseed = np.zeros((R, M), np.int8)
            for r in range(R):
                rr = np.random.default_rng(1000 * seed + 100 * r + n)
                z, u = sample_x(rr, n); y = (rr.random(n) < eta_of_z(z, eps)[0]).astype(int)
                kind, m = mk()
                if kind == "custom": m.fit(z, u, y); P[r] = m.predict(zt, ut)
                else:
                    if hasattr(m, "random_state"): m.set_params(random_state=r)
                    P[r] = m.fit(feats(z, u), y).predict(Xt)
                if r == 0: z0, u0, y0 = z, u, y
            for r in range(R):                                     # same sample, different seeds
                kind, m = mk()
                if kind == "custom": m.fit(z0, u0, y0); Pseed[r] = m.predict(zt, ut)
                else:
                    if hasattr(m, "random_state"): m.set_params(random_state=r)
                    Pseed[r] = m.fit(feats(z0, u0), y0).predict(Xt)
            def stats(P):
                p1 = P.mean(0); pi = np.minimum(p1, 1 - p1)
                risk = float(np.mean(np.where(P == 1, 1 - e1, e1)))            # exact expected noisy-label error
                dis = float(np.mean(2 * p1 * (1 - p1) * R / (R - 1)))          # unbiased pairwise disagreement
                V = float(np.mean(np.where(pi > 0, h2(pi), 0)))
                return risk, dis, V, float(np.mean(pi))
            risk, dis, V, pib = stats(P); risk_s, dis_s, V_s, _ = stats(Pseed)
            out.append(dict(n=n, learner=name, Rstar=Rstar, risk=risk, excess=risk - Rstar, dis=dis, V=V, pibar=pib,
                            dis_seed=dis_s, risk_seed=risk_s, bias=float(np.mean(np.abs(2 * e1 - 1) * ((P.mean(0) >= .5) != b_t)))))
            print(json.dumps(out[-1]), flush=True)
    json.dump(out, open("t_disagreement.json", "w"), indent=1)
if __name__ == "__main__":
    run()
