"""Bootstrap confidence intervals for the headline rates.

Reads item-level records from results/*.json and emits results/bootstrap_ci.json
with percentile bootstrap CIs (B=10,000) for the swap-experiment rates quoted
in the write-up, plus paired differences where the comparison is the claim.

Method notes:
- Single rates and paired differences use the ordinary percentile bootstrap
  over items. Items are independent facts/prompts, so item resampling is the
  right unit.
- Each experiment block draws from its own deterministic RNG (seeded from the
  block label), so results do not depend on which other blocks run.

Run:  .venv/bin/python experiments/stats/run_bootstrap.py
"""

import json
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
B = 10_000
SEED = 0
MODELS = ["qwen17b", "qwen4b"]


def block_rng(label: str) -> np.random.Generator:
    """Deterministic per-block generator: independent of block ordering."""
    return np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(label.encode())]))


def ci(values: np.ndarray, rng: np.random.Generator) -> dict:
    """Percentile bootstrap CI for the mean of a 0/1 (or real) item vector."""
    n = len(values)
    idx = rng.integers(0, n, size=(B, n))
    means = values[idx].mean(axis=1)
    return {
        "est": float(values.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": n,
    }


def load(name: str) -> dict:
    return json.load(open(RESULTS / name))


def paired_hits_block(name: str, model: str, arms: list[str]) -> dict:
    d = load(f"{name}_{model}.json")
    recs = [r for r in d["records"] if r["baseline_ok"]]
    rng = block_rng(f"{name}_{model}")
    out = {arm: ci(np.array([r[f"{arm}_hit"] for r in recs], float), rng) for arm in arms}
    if len(arms) >= 2:
        a0, a1 = arms[0], arms[1]
        out[f"{a1}_minus_{a0}"] = ci(
            np.array([float(r[f"{a1}_hit"]) - float(r[f"{a0}_hit"]) for r in recs]), rng
        )
    return out


def main() -> None:
    report = {"seed": SEED, "B": B, "models": {}}
    for model in MODELS:
        report["models"][model] = {
            # E2: arm a = intermediate-entity swap, arm b = answer-token control
            "e2": paired_hits_block("e2", model, ["a", "b"]),
            # E2p: same arms with mass-mean probe directions
            "e2p": paired_hits_block("e2p", model, ["a", "b"]),
            # E2t: same arms with trained-probe directions
            "e2t": paired_hits_block("e2t", model, ["a", "b"]),
            # E5: b = answer source, c = intermediate source, d = absent word
            "e5": paired_hits_block("e5", model, ["c", "b", "d"]),
        }
    out_path = RESULTS / "bootstrap_ci.json"
    json.dump(report, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")
    for model in MODELS:
        m = report["models"][model]
        for exp in ("e2", "e2t", "e5"):
            for key, blk in m[exp].items():
                if "minus" in key:
                    print(
                        f"  {model} {exp} {key}: {blk['est']:+.1%} "
                        f"[{blk['lo']:+.1%}, {blk['hi']:+.1%}] (n={blk['n']})"
                    )


if __name__ == "__main__":
    main()
