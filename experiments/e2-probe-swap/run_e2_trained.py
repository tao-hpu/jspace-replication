"""E2t: rerun the E2 arms along *trained* probe directions.

Closes the last open branch of the E2/E2p direction-source question. E2 used
Jacobian-lens directions, E2p the closed-form mass-mean stand-in; the official
experiment swaps along trained linear-probe directions. Here each band layer
gets a multinomial logistic probe over the entity pool (entities as classes,
inputs are final-token residuals over neutral template contexts), and the
direction for entity X at layer l is the normalized class-weight row w_X(l).

Probe quality is sanity-checked before use: probes are refit with 4 templates
held out and must classify the held-out contexts well above chance; the
held-out top-1 accuracy per layer is stored in the results file. Swap
mechanism, band, positions, grading, and items are identical to E2/E2p, so
any change in the A-vs-B gap is attributable to the direction source alone.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e2-probe-swap/run_e2_trained.py [1.7b|4b]
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.interventions import DirectionSwapHooks  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import MODELS, greedy_gen, text_match  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e2-probe-swap"))
from run_e2 import mcnemar_exact_p  # noqa: E402
from run_e2_probe import TEMPLATES as BASE_TEMPLATES  # noqa: E402

DATA = ROOT / "third_party/jacobian-lens/data/experiments/probe-swap.json"

# 12 extra frames on top of the E2p set: 12 positives per class is thin for a
# trained probe, 24 keeps the fit stable under weight decay.
EXTRA_TEMPLATES = [
    "A short report on {}",
    "They asked me a question about {}",
    "The book's final chapter is about {}",
    "We spent the meeting discussing {}",
    "My notes from class are about {}",
    "The exhibit downstairs features {}",
    "She gave a talk on {}",
    "The trivia answer turned out to be {}",
    "In the end it was all about {}",
    "The riddle's solution is {}",
    "Today's topic is {}",
    "The subject of the painting is {}",
]
TEMPLATES = BASE_TEMPLATES + EXTRA_TEMPLATES
N_HELDOUT = 4  # templates reserved for the quality check refit


@torch.no_grad()
def entity_acts(hf, tok, entities: list[str], band: list[int], device) -> torch.Tensor:
    """[n_ent, n_tpl, len(band), d] final-token residuals, float32 CPU."""
    out = []
    for i, ent in enumerate(entities):
        per_tpl = []
        for tpl in TEMPLATES:
            ids = tok(tpl.format(ent), return_tensors="pt").to(device)
            hs = hf(**ids, output_hidden_states=True).hidden_states
            per_tpl.append(torch.stack([hs[l + 1][0, -1].float() for l in band]).cpu())
        out.append(torch.stack(per_tpl))
        if (i + 1) % 25 == 0:
            print(f"  activations: {i + 1}/{len(entities)} entities")
    return torch.stack(out)


def fit_probe(x: torch.Tensor, y: torch.Tensor, n_classes: int, device,
              steps: int = 300, lr: float = 0.05, wd: float = 1e-3) -> torch.Tensor:
    """Full-batch multinomial logistic regression; returns W [n_classes, d]."""
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x, y = x.to(device), y.to(device)
    w = torch.zeros(n_classes, x.shape[1], device=device, requires_grad=True)
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w.T, y)
        loss.backward()
        opt.step()
    return w.detach().cpu()


def main(model_key: str = "4b") -> None:
    model_id, _, _ = MODELS[model_key]
    prior = json.loads((ROOT / "results" / f"e2_qwen{model_key.replace('.', '')}.json").read_text())
    e2p = json.loads((ROOT / "results" / f"e2p_qwen{model_key.replace('.', '')}.json").read_text())
    band = list(range(prior["band"][0], prior["band"][1] + 1))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    print(f"model={model_id} band={band[0]}..{band[-1]}")

    items = json.load(open(DATA))["items"]
    lens_by_name = {r["name"]: r for r in prior["records"]}
    e2p_by_name = {r["name"]: r for r in e2p["records"]}

    entities = sorted({it[k] for it in items for k in ("intermediate", "swap_to", "answer", "swap_answer")})
    n_ent, n_tpl = len(entities), len(TEMPLATES)
    print(f"collecting activations: {n_ent} entities x {n_tpl} templates")
    acts = entity_acts(hf, tok, entities, band, device)  # [E, T, L, d]
    labels = torch.arange(n_ent).repeat_interleave(n_tpl)

    print("fitting probes per band layer (with held-out quality check)")
    dirs: dict[str, dict[int, torch.Tensor]] = {e: {} for e in entities}
    heldout_acc = {}
    for j, l in enumerate(band):
        x_all = acts[:, :, j, :].reshape(n_ent * n_tpl, -1)
        # quality check: refit without the last N_HELDOUT templates
        train_mask = (torch.arange(n_tpl) < n_tpl - N_HELDOUT).repeat(n_ent)
        w_chk = fit_probe(x_all[train_mask], labels[train_mask], n_ent, device)
        x_h = x_all[~train_mask]
        x_h = (x_h - x_h.mean(0)) / (x_h.std(0) + 1e-6)
        acc = float((x_h @ w_chk.T).argmax(1).eq(labels[~train_mask]).float().mean())
        heldout_acc[str(l)] = acc
        # final directions: fit on all templates
        w = fit_probe(x_all, labels, n_ent, device)
        for e_idx, ent in enumerate(entities):
            d = w[e_idx]
            dirs[ent][l] = (d / d.norm()).to(device)
        print(f"  L{l}: held-out top-1 {acc:.1%} (chance {1 / n_ent:.1%})")

    records = []
    for it in items:
        prompt = it["prompt"].rstrip()
        pair = lambda a, b: {l: (dirs[a][l], dirs[b][l]) for l in band}  # noqa: E731

        with_a = DirectionSwapHooks(model.layers, pair(it["intermediate"], it["swap_to"]))
        _, a_text = greedy_gen(hf, tok, prompt, device, swap_ctx=with_a)

        with_b = DirectionSwapHooks(model.layers, pair(it["answer"], it["swap_answer"]))
        _, b_text = greedy_gen(hf, tok, prompt, device, swap_ctx=with_b)

        lens_rec, e2p_rec = lens_by_name[it["name"]], e2p_by_name[it["name"]]
        records.append({
            "name": it["name"], "category": it["category"],
            "baseline_ok": lens_rec["baseline_ok"],
            "a_text": a_text, "b_text": b_text,
            "a_hit": text_match(a_text, it["swap_answer"]),
            "b_hit": text_match(b_text, it["swap_answer"]),
            "a_stayed": text_match(a_text, it["answer"]),
            "b_stayed": text_match(b_text, it["answer"]),
            "lens_a_hit": lens_rec["a_hit"], "lens_b_hit": lens_rec["b_hit"],
            "massmean_a_hit": e2p_rec["a_hit"], "massmean_b_hit": e2p_rec["b_hit"],
        })

    ok = [r for r in records if r["baseline_ok"]]
    rate = lambda key: sum(r[key] for r in ok) / len(ok)  # noqa: E731

    def mcnemar(k1: str, k2: str) -> tuple[int, int, float]:
        only1 = sum(r[k1] and not r[k2] for r in ok)
        only2 = sum(r[k2] and not r[k1] for r in ok)
        return only1, only2, mcnemar_exact_p(only1, only2)

    ab = mcnemar("a_hit", "b_hit")
    a_vs_lens = mcnemar("a_hit", "lens_a_hit")
    a_vs_mm = mcnemar("a_hit", "massmean_a_hit")

    print(f"\nbaseline correct (from E2): {len(ok)}/{len(records)}")
    print(f"arm A'' (trained intermediate) hit: {rate('a_hit'):.1%}   stayed: {rate('a_stayed'):.1%}")
    print(f"arm B'' (trained answer ctrl)  hit: {rate('b_hit'):.1%}   stayed: {rate('b_stayed'):.1%}")
    print(f"A'' vs B'': only-A {ab[0]}, only-B {ab[1]}, exact McNemar p = {ab[2]:.4f}")
    print(f"A trained vs lens:      gained {a_vs_lens[0]}, lost {a_vs_lens[1]}, p = {a_vs_lens[2]:.4f}")
    print(f"A trained vs mass-mean: gained {a_vs_mm[0]}, lost {a_vs_mm[1]}, p = {a_vs_mm[2]:.4f}")
    print(f"(lens arms: A {rate('lens_a_hit'):.1%} B {rate('lens_b_hit'):.1%}; "
          f"mass-mean arms: A {rate('massmean_a_hit'):.1%} B {rate('massmean_b_hit'):.1%})")

    out = ROOT / "results" / f"e2t_qwen{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({
        "model": model_id, "band": [band[0], band[-1]],
        "n_templates": n_tpl, "n_heldout": N_HELDOUT,
        "probe_heldout_top1": heldout_acc,
        "records": records,
        "summary": {"n_ok": len(ok), "a_hit": rate("a_hit"), "b_hit": rate("b_hit"),
                    "p_a_vs_b": ab[2],
                    "p_a_trained_vs_lens": a_vs_lens[2],
                    "p_a_trained_vs_massmean": a_vs_mm[2]},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4b")
