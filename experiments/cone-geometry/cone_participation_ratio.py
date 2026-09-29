"""Transport geometry of Jacobian-lens directions: uncentered vs centered
participation ratio (effective dimensionality).

For each model we take the single-token entity words of the official
probe-swap set, and at the middle band layer build two sets of unit
directions:

    raw          d_t = normalize(W_U[t])
    transported  d_t = normalize(J_L^T W_U[t])

and report, for each set:

    mean_cos       mean pairwise cosine (off-diagonal Gram entries)
    pr_uncentered  participation ratio of the unit rows as given
                   (the quantity used in earlier drafts)
    pr_centered    participation ratio after subtracting the mean direction

Why both: for n unit vectors the uncentered PR equals
n^2 / ||G||_F^2 = 1 / ((n-1)/n * mean(cos^2) + 1/n), so it is bounded by
roughly 1/c^2 where c is the mean cosine. A set that shares one common
direction therefore has a small uncentered PR regardless of how its
residual variation is spread. The centered PR removes that shared component
and measures the spread of what remains.

No model forward pass is run: only the unembedding matrix and the fitted
lens Jacobian of each model are loaded (CPU is sufficient; the 27B/32B
lenses are large downloads).

Note on Qwen3-32B: the lens file for this model was published as an
unfinalized fit() checkpoint (running Jacobian sum and n_done). The loader
below finalizes it as jacobian_sum / n_done. An earlier archived run of the
uncentered statistic reported mean cosine 0.395 and PR 5.2 for this model;
the current file gives 0.411 and 4.89. The paper reports the values
produced by this script for every model.

Run:
    HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/cone-geometry/cone_participation_ratio.py [model_label ...]
Output: results/cone_pr_centered.json (appends; models already present are skipped).
"""

from __future__ import annotations

import gc
import json
import pathlib
import sys

import torch
import transformers
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party/jacobian-lens/data/experiments/probe-swap.json"
OUT = ROOT / "results/cone_pr_centered.json"

# (label, tokenizer/model id, (lens repo or local path, lens file), unembedding tensor names)
MODELS = [
    ("pythia-70m", "EleutherAI/pythia-70m-deduped",
     ("neuronpedia/jacobian-lens", "pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt"),
     ["embed_out.weight"]),
    # self-trained 124M GPT-2-style control with a lens fitted by us (n=150 prompts)
    ("selftrained-124m", str(ROOT / "out/selftrained-124m-hf"),
     (str(ROOT / "out/selftrained-124m_jacobian_lens.pt"), None),
     ["wte.weight", "transformer.wte.weight"]),
    ("gpt2-124m", "gpt2",
     ("neuronpedia/jacobian-lens", "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"),
     ["wte.weight", "transformer.wte.weight"]),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B",
     ("neuronpedia/jacobian-lens", "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"),
     ["lm_head.weight", "model.embed_tokens.weight"]),
    ("qwen3-4b", "Qwen/Qwen3-4B",
     ("neuronpedia/jacobian-lens", "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"),
     ["lm_head.weight", "model.embed_tokens.weight"]),
    ("qwen3-8b", "Qwen/Qwen3-8B",
     ("neuronpedia/jacobian-lens", "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"),
     ["lm_head.weight", "model.embed_tokens.weight"]),
    ("qwen3-14b", "Qwen/Qwen3-14B",
     ("neuronpedia/jacobian-lens", "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"),
     ["lm_head.weight", "model.embed_tokens.weight"]),
    ("qwen3-32b", "Qwen/Qwen3-32B",
     ("neuronpedia/jacobian-lens", "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"),
     ["lm_head.weight", "model.embed_tokens.weight"]),
    ("qwen3.6-27b", "Qwen/Qwen3.6-27B",
     ("neuronpedia/jacobian-lens", "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"),
     ["lm_head.weight", "model.embed_tokens.weight"]),
]


def load_unembed(repo: str, names: list[str]) -> torch.Tensor:
    """Load only the unembedding tensor (selective safetensors read)."""
    local = pathlib.Path(repo)
    if local.is_dir():
        for f in sorted(local.glob("*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for n in names:
                    if n in sf.keys():
                        return sf.get_tensor(n).float()
        raise KeyError(f"{names} not found in {repo}")
    try:
        index_path = hf_hub_download(repo, "model.safetensors.index.json")
    except EntryNotFoundError:
        index_path = None
    if index_path is None:          # unsharded repo: a single model.safetensors
        with safe_open(hf_hub_download(repo, "model.safetensors"), framework="pt") as sf:
            for n in names:
                if n in sf.keys():
                    return sf.get_tensor(n).float()
        raise KeyError(f"{names} not found in {repo}")
    wmap = json.load(open(index_path))["weight_map"]
    for n in names:
        if n in wmap:
            with safe_open(hf_hub_download(repo, wmap[n]), framework="pt") as sf:
                return sf.get_tensor(n).float()
    raise KeyError(f"{names} not in weight_map of {repo}")


def load_lens(lens_repo: str, lens_file: str | None):
    """JacobianLens.from_pretrained, falling back to finalizing a raw fit()
    checkpoint (J = jacobian_sum / n_done) when the file is not finalized."""
    try:
        if lens_file is None:
            return jlens.JacobianLens.from_pretrained(lens_repo)
        return jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    except ValueError:
        if lens_file is None:       # local lens directory: nothing to download, surface the real error
            raise
        state = torch.load(hf_hub_download(lens_repo, lens_file), map_location="cpu", weights_only=True)
        jac = {int(l): (state["jacobian_sum"][l] / state["n_done"]).float() for l in state["jacobian_sum"]}
        d_model = next(iter(jac.values())).shape[0]
        return jlens.JacobianLens(jacobians=jac, n_prompts=state["n_done"], d_model=d_model)


def participation_ratio(X: torch.Tensor) -> float:
    s = torch.linalg.svdvals(X.double())
    return float((s.square().sum() ** 2) / s.pow(4).sum())


def stats(d: torch.Tensor) -> dict:
    """d: [n, dim] unit rows."""
    n = d.shape[0]
    off = (d @ d.T)[~torch.eye(n, dtype=torch.bool)]
    c = float(off.mean())
    return dict(n=n, mean_cos=c,
                inv_c2=1 / c ** 2 if c > 0 else None,
                pr_uncentered=participation_ratio(d),
                pr_centered=participation_ratio(d - d.mean(0, keepdim=True)))


def main(only: list[str] | None = None) -> None:
    res = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r["model"] for r in res}
    entities = sorted({it[k] for it in json.load(open(DATA))["items"]
                       for k in ("intermediate", "swap_to", "answer", "swap_answer")})
    for label, tok_id, (lens_repo, lens_file), names in MODELS:
        if label in done or (only and label not in only):
            continue
        tok = transformers.AutoTokenizer.from_pretrained(tok_id)
        lens = load_lens(lens_repo, lens_file)
        W = load_unembed(tok_id, names)
        toks = sorted({tok.encode(" " + w, add_special_tokens=False)[0] for w in entities
                       if len(tok.encode(" " + w, add_special_tokens=False)) == 1})
        layer = lens.source_layers[len(lens.source_layers) // 2]
        J = lens.jacobians[layer].float()
        U = W[toks]
        row = dict(model=label, layer=int(layer),
                   raw=stats(torch.nn.functional.normalize(U, dim=1)),
                   jlens=stats(torch.nn.functional.normalize(U @ J, dim=1)))
        res.append(row)
        OUT.write_text(json.dumps(res, indent=2))
        r, j = row["raw"], row["jlens"]
        inv = "n/a" if j["inv_c2"] is None else f"{j['inv_c2']:.2f}"
        print(f"{label:16s} n={j['n']} raw: c={r['mean_cos']:.3f} PRunc={r['pr_uncentered']:.1f} "
              f"PRcen={r['pr_centered']:.1f} | J: c={j['mean_cos']:.3f} 1/c2={inv} "
              f"PRunc={j['pr_uncentered']:.2f} PRcen={j['pr_centered']:.2f}", flush=True)
        del lens, W, J
        gc.collect()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
