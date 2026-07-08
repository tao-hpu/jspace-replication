"""Aggregate the multi-seed hardening runs into seed x item variance.

Reads the canonical E6/E7 result files plus every seed-suffixed sibling and
reports each headline rate as canonical value + mean +/- sd across seeds.

What each family's variance means (they are NOT symmetric, and the paper must
say so):
  E6  the register axis is re-estimated on a bootstrap resample of the parallel
      pairs (E6_RESAMPLE=1) and the random control is reseeded, so the register
      flip rates carry genuine estimation variance.
  E7  only the randdir specificity control varies; the none/full/half arms are
      deterministic under greedy decoding, so the headline capture rates carry
      no seed variance. These runs only harden that the randdir null is robust.

Run:  .venv/bin/python experiments/stats/aggregate_multiseed.py
"""

from __future__ import annotations

import glob
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def _msd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = st.mean(xs)
    s = st.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s, min(xs), max(xs)


def _fmt(msd):
    if msd is None:
        return "  (no seeds)"
    m, s, lo, hi = msd
    return f"{m:6.1%} +/- {s:4.1%}  [min {lo:.1%}, max {hi:.1%}]"


def agg_e6(stem: str):
    canon = RES / f"{stem}.json"
    if not canon.exists():
        return
    seeds = sorted(glob.glob(str(RES / f"{stem}_seed*r.json")))
    print(f"\n=== E6 {stem}  (canonical + {len(seeds)} resample seeds) ===")
    cj = json.loads(canon.read_text())
    alphas = [f"{a:g}" for a in cj["alphas"]]
    sjs = [json.loads(pathlib.Path(p).read_text()) for p in seeds]
    for a in alphas:
        for key in ("register_flip_full", "register_preserved", "random_flip_full"):
            cval = cj["summary"]["per_alpha"][a][key]
            svals = [s["summary"]["per_alpha"].get(a, {}).get(key) for s in sjs]
            print(f"  a={a:6s} {key:22s} canonical {cval:6.1%} | seeds {_fmt(_msd(svals))}")


def agg_e7(stem: str):
    canon = RES / f"{stem}.json"
    if not canon.exists():
        return
    seeds = sorted(glob.glob(str(RES / f"{stem}_seed*.json")))
    print(f"\n=== E7 {stem}  (canonical + {len(seeds)} randdir seeds) ===")
    cj = json.loads(canon.read_text())
    sjs = [json.loads(pathlib.Path(p).read_text()) for p in seeds]
    for key in ("flip", "restate_swapped"):
        cval = cj["summary"]["arms"]["randdir"][key]
        svals = [s["summary"]["arms"]["randdir"].get(key) for s in sjs]
        print(f"  randdir {key:16s} canonical {cval:6.1%} | seeds {_fmt(_msd(svals))}")


if __name__ == "__main__":
    for stem in ("e6_covert_register_qwen17b", "e6_covert_register_qwen4b"):
        agg_e6(stem)
    for stem in ("e7_perspectival_qwen17b", "e7_perspectival_qwen4b"):
        agg_e7(stem)
