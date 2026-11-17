"""Protocol B, additional seeds: python r6_seeds.py 1 2 3 4  (skips cases already on disk)."""
import sys
from r6_cost import run_case, NEW, OLD
seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4]
for seed in seeds:
    for ds in NEW + OLD:
        for kind in ("sym0.2", "classdep", "instdep"):
            run_case(ds, kind, seed)
