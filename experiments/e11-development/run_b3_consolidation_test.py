"""Statistical test for the flicker-vs-lock contrast (1B vs 1.4B intermediates).

The paper's headline claim is that multi-hop intermediate readability opens at
the same epoch at 1B and 1.4B but consolidates only at 1.4B. This script backs
that contrast with a paired bootstrap over the 29 multi-hop probe items, using
the per-item ranks stored in results/b3_probe_rank_*.json.

Two statistics over the late-training window (steps >= LATE_START, i.e. after
both models have registered onset):

  1. endpoint separation: log10(median intermediate rank) at step 143000,
     1B minus 1.4B. CI excluding 0 -> the endpoint gap exceeds probe-item noise.
  2. excess path length (flicker): total variation of log10(median rank)
     across late checkpoints minus the absolute net change, 1B minus 1.4B.
     A monotone trajectory scores 0 regardless of how fast it declines; an
     oscillating one scores the summed size of its reversals. CI excluding 0
     -> the 1B trajectory genuinely oscillates more (flicker vs lock).
     (A plain SD of log-medians cannot separate the two: a steady decline
     and an oscillation of similar range have similar SDs.)

The bootstrap is paired: each replicate resamples the same item indices for
both models, since both are evaluated on the identical probe set.

Run:  .venv/bin/python experiments/e11-development/run_b3_consolidation_test.py
Out:  results/b3_consolidation_test.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["pythia-1b-deduped", "pythia-1.4b-deduped"]
LATE_START = 13000
ENDPOINT = 143000
N_BOOT = 2000
SEED = 0


def load_late_ranks(model: str) -> tuple[list[int], np.ndarray]:
    data = json.loads((ROOT / f"results/b3_probe_rank_{model}.json").read_text())
    rows = [r for r in data["rows"] if r["step"] >= LATE_START]
    steps = [r["step"] for r in rows]
    ranks = np.array([r["intermediate"]["ranks"] for r in rows], dtype=float)
    return steps, ranks  # shape (n_late_ckpts, n_items)


def stats_for(ranks: np.ndarray, item_idx: np.ndarray) -> tuple[float, float]:
    med = np.median(ranks[:, item_idx], axis=1)
    log_med = np.log10(med)
    total_variation = float(np.sum(np.abs(np.diff(log_med))))
    net_change = float(np.abs(log_med[-1] - log_med[0]))
    return float(log_med[-1]), total_variation - net_change


def main() -> None:
    steps_1b, ranks_1b = load_late_ranks(MODELS[0])
    steps_14b, ranks_14b = load_late_ranks(MODELS[1])
    assert steps_1b == steps_14b and steps_1b[-1] == ENDPOINT
    n_items = ranks_1b.shape[1]
    full = np.arange(n_items)

    obs_end_1b, obs_sd_1b = stats_for(ranks_1b, full)
    obs_end_14b, obs_sd_14b = stats_for(ranks_14b, full)

    rng = np.random.default_rng(SEED)
    end_diffs, sd_diffs = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_items, size=n_items)
        e1, s1 = stats_for(ranks_1b, idx)
        e2, s2 = stats_for(ranks_14b, idx)
        end_diffs.append(e1 - e2)
        sd_diffs.append(s1 - s2)
    end_diffs = np.array(end_diffs)
    sd_diffs = np.array(sd_diffs)

    def ci(a: np.ndarray) -> list[float]:
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out = {
        "models": MODELS,
        "late_steps": steps_1b,
        "n_items": n_items,
        "n_boot": N_BOOT,
        "seed": SEED,
        "observed": {
            "endpoint_median_rank": {
                MODELS[0]: float(np.median(ranks_1b[-1])),
                MODELS[1]: float(np.median(ranks_14b[-1])),
            },
            "endpoint_log10_median_diff": obs_end_1b - obs_end_14b,
            "excess_path_length": {
                MODELS[0]: obs_sd_1b,
                MODELS[1]: obs_sd_14b,
            },
            "excess_path_length_diff": obs_sd_1b - obs_sd_14b,
        },
        "bootstrap": {
            "endpoint_log10_median_diff_ci95": ci(end_diffs),
            "excess_path_length_diff_ci95": ci(sd_diffs),
            "p_endpoint_diff_le_0": float(np.mean(end_diffs <= 0)),
            "p_excess_path_diff_le_0": float(np.mean(sd_diffs <= 0)),
        },
    }
    dest = ROOT / "results/b3_consolidation_test.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["observed"], indent=2))
    print(json.dumps(out["bootstrap"], indent=2))
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
