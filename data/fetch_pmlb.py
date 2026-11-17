"""Download the ten PMLB datasets used in Protocol B (main Sec. 7.5) into data/pmlb/<name>/<name>.tsv.gz,
and the nine PMLB regression datasets of Supp. Sec. J into data/pmlb_reg/<name>/<name>.tsv.gz.
Uses the PMLB repository's LFS media URL (raw.githubusercontent.com serves LFS pointers, not data).
Behind a TLS-intercepting proxy, set REQUESTS_CA_BUNDLE to your CA bundle."""
import os, sys, urllib.request
NAMES = ["twonorm", "ring", "magic", "phoneme", "spambase", "churn", "waveform_40", "satimage", "optdigits", "kr_vs_kp"]
# regression datasets for Supp. Sec. J (written to data/pmlb_reg/)
REG = ["1199_BNG_echoMonths", "215_2dplanes", "225_puma8NH", "197_cpu_act", "503_wind", "529_pollen", "537_houses", "564_fried", "344_mv"]
BASE = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pmlb")
ca = os.environ.get("REQUESTS_CA_BUNDLE")
ctx = None
if ca:
    import ssl; ctx = ssl.create_default_context(cafile=ca)
for n in NAMES + REG:
    d = os.path.join(out if n in NAMES else out + "_reg", n); os.makedirs(d, exist_ok=True); f = os.path.join(d, n + ".tsv.gz")
    if os.path.exists(f) and os.path.getsize(f) > 1000:
        print("have", n); continue
    with urllib.request.urlopen(f"{BASE}/{n}/{n}.tsv.gz", context=ctx) as r, open(f, "wb") as w:
        w.write(r.read())
    print("fetched", n, os.path.getsize(f), "bytes")
