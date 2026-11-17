"""B3 follow-up: uncertainty and layer sensitivity for the geometry curves.

Two gaps in the first B3 pass: the workspace-defining quantities (centered
effective dimension of the transported/raw cones, coordinate drift) were
reported as point estimates, and the transported-cone dimension was read at a
single mid layer whose choice was never stress-tested. Both are pure linear
algebra on the stored per-checkpoint lenses, so this script closes them
without any refitting:

  1. uncertainty: jackknife (leave-one-entity-out) SEs and normal 95% CIs
     for the centered eff_dim of both cones and their raw/transported ratio.
     Jackknife, not bootstrap: resampling entities with replacement inserts
     duplicate rows, which mechanically lowers a participation ratio, so the
     bootstrap distribution is biased below the observed value (verified: the
     point estimate landed outside its own percentile CI). Leave-one-out has
     no duplicates and no such bias. Drift summaries (plain means over the
     512-token vocab sample) keep an ordinary 2,000-resample bootstrap.
  2. layer profile: centered eff_dim of the transported cone at every fitted
     source layer, for the layer-sensitivity appendix.

Run (on the GPU box, from the repo root):
  python experiments/e11-development/run_b3_effdim_ci.py --model EleutherAI/pythia-1.4b-deduped
Out:  results/b3_effdim_ci_{tag}.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
LENS_DIR = ROOT / "out" / "_b2_lenses"
DATA = ROOT / "third_party/jacobian-lens/data/experiments/probe-swap.json"
N_BOOT = 2000
SEED = 0


def _pr(mat: torch.Tensor) -> float:
    s = torch.linalg.svdvals(mat)
    return float((s.square().sum() ** 2) / (s.pow(4).sum()))


def _pr_centered(dirs: torch.Tensor) -> float:
    return _pr(dirs - dirs.mean(0, keepdim=True))


def entity_token_ids(tok) -> list[int]:
    entities = sorted({it[k] for it in json.load(open(DATA))["items"]
                       for k in ("intermediate", "swap_to", "answer", "swap_answer")})
    ids = []
    for w in entities:
        enc = tok.encode(" " + w, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return sorted(set(ids))


def step_of(rev: str) -> int:
    return int(rev.replace("step", "").replace(".pt", ""))


def boot_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


@torch.no_grad()
def analyze(model_id: str, revision: str, tok, entity_ids: list[int],
            rng: np.random.Generator) -> dict:
    lens = jlens.JacobianLens.from_pretrained(
        str(LENS_DIR / f"{model_id.split('/')[-1]}_{revision}.pt"))
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.float32)
    W = hf.get_output_embeddings().weight.float()
    vocab, _ = W.shape
    n_layers = hf.config.num_hidden_layers
    del hf

    layers = lens.source_layers
    mid = layers[len(layers) // 2]
    U = W[entity_ids]
    raw_dirs = torch.nn.functional.normalize(U, dim=1)
    jl_dirs = torch.nn.functional.normalize(U @ lens.jacobians[mid].float(), dim=1)
    n_ent = raw_dirs.shape[0]

    # 1a. jackknife over entities for the centered eff_dims and their ratio
    # (bootstrap is biased low for participation ratios: duplicated rows
    # concentrate the spectrum; leave-one-out has no duplicates)
    jl_i, raw_i = [], []
    for i in range(n_ent):
        keep = torch.arange(n_ent) != i
        jl_i.append(_pr_centered(jl_dirs[keep]))
        raw_i.append(_pr_centered(raw_dirs[keep]))
    jl_i = np.array(jl_i)
    raw_i = np.array(raw_i)
    ratio_i = raw_i / jl_i
    obs_jl = _pr_centered(jl_dirs)
    obs_raw = _pr_centered(raw_dirs)

    def jack(theta_i: np.ndarray, theta_hat: float) -> tuple[float, list[float]]:
        se = float(np.sqrt((n_ent - 1) / n_ent
                           * ((theta_i - theta_i.mean()) ** 2).sum()))
        return se, [theta_hat - 1.96 * se, theta_hat + 1.96 * se]

    jl_se, jl_ci = jack(jl_i, obs_jl)
    raw_se, raw_ci = jack(raw_i, obs_raw)
    ratio_se, ratio_ci = jack(ratio_i, obs_raw / obs_jl)

    # 1b. vocab bootstrap for the drift summaries (cos values reweighted)
    vocab_ids = list(range(0, vocab, max(1, vocab // 512)))[:512]
    Uv = W[vocab_ids]
    per_layer_cos = []
    for l in layers:
        J = lens.jacobians[l].float()
        per_layer_cos.append(
            torch.nn.functional.cosine_similarity(Uv @ J, Uv, dim=1).numpy())
    per_layer_cos = np.stack(per_layer_cos)          # [n_layers_fit, n_vocab]
    deep_mask = np.array([l / n_layers >= 0.66 for l in layers])
    n_voc = per_layer_cos.shape[1]
    mean_b, deep_b = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_voc, size=n_voc)
        sub = per_layer_cos[:, idx].mean(axis=1)
        mean_b.append(float(sub.mean()))
        deep_b.append(float(sub[deep_mask].mean()) if deep_mask.any() else None)
    obs_layer_mean = per_layer_cos.mean(axis=1)

    # 2. layer profile of the transported centered eff_dim
    profile = []
    for l in layers:
        d = torch.nn.functional.normalize(U @ lens.jacobians[l].float(), dim=1)
        profile.append({"layer": int(l), "frac_depth": l / n_layers,
                        "eff_dim_jl_centered": _pr_centered(d)})

    return {
        "step": step_of(revision), "mid_layer": int(mid), "n_layers": n_layers,
        "eff_dim_jl_centered": obs_jl,
        "eff_dim_jl_centered_se": jl_se,
        "eff_dim_jl_centered_ci95": jl_ci,
        "eff_dim_raw_centered": obs_raw,
        "eff_dim_raw_centered_se": raw_se,
        "eff_dim_raw_centered_ci95": raw_ci,
        "raw_over_jl_ratio": obs_raw / obs_jl,
        "raw_over_jl_ratio_se": ratio_se,
        "raw_over_jl_ratio_ci95": ratio_ci,
        "mean_drift": float(obs_layer_mean.mean()),
        "mean_drift_ci95": boot_ci(np.array(mean_b)),
        "deep_drift": float(obs_layer_mean[deep_mask].mean()) if deep_mask.any() else None,
        "deep_drift_ci95": boot_ci(np.array([x for x in deep_b if x is not None]))
        if deep_mask.any() else None,
        "layer_profile": profile,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    entity_ids = entity_token_ids(tok)
    rng = np.random.default_rng(SEED)

    revs = sorted((p.name.replace(f"{tag}_", "") for p in
                   LENS_DIR.glob(f"{tag}_step*.pt")), key=step_of)
    print(f"model={args.model} entities={len(entity_ids)} checkpoints={len(revs)}",
          flush=True)

    rows = []
    for rev in revs:
        r = analyze(args.model, rev.replace(".pt", ""), tok, entity_ids, rng)
        rows.append(r)
        print(f"  step {r['step']:>7d} jl={r['eff_dim_jl_centered']:6.1f} "
              f"CI[{r['eff_dim_jl_centered_ci95'][0]:6.1f},{r['eff_dim_jl_centered_ci95'][1]:6.1f}] "
              f"ratio={r['raw_over_jl_ratio']:.2f} "
              f"CI[{r['raw_over_jl_ratio_ci95'][0]:.2f},{r['raw_over_jl_ratio_ci95'][1]:.2f}]",
              flush=True)

    dst = ROOT / "results" / f"b3_effdim_ci_{tag}.json"
    dst.write_text(json.dumps(
        {"model": args.model, "entity_ci_method": "jackknife",
         "drift_ci_method": "bootstrap", "n_boot": N_BOOT, "seed": SEED,
         "rows": rows}, indent=1))
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
