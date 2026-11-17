"""Smoke test: does the B chain (jlens.from_hf -> fit -> curves stats -> probe readout)
run end-to-end on OlmoForCausalLM? Uses a RANDOM-INIT model built from the real OLMo-1B-0724
config (no 5 GB download needed), 2 prompts x 32 tokens. Numbers are meaningless; only the
code path is under test. Run on the Mac:
  .venv/bin/python experiments/e11-development/smoke_olmo_chain.py --config ~/ckpts/olmo-1b-0724/tokenizer/config.json
"""
from __future__ import annotations
import argparse, json, pathlib, time, os
import torch, transformers
import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party/jacobian-lens/data/experiments/probe-swap.json"

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.expanduser("~/ckpts/olmo-1b-0724/tokenizer/config.json"))
ap.add_argument("--tok", default=os.path.expanduser("~/ckpts/olmo-1b-0724/tokenizer"))
ap.add_argument("--layers", type=int, default=0, help="shrink num_hidden_layers for speed (0 = real 16)")
args = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = transformers.AutoConfig.from_pretrained(args.config)
if args.layers:
    cfg.num_hidden_layers = args.layers
torch.manual_seed(0)
hf = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).to(dev).eval()
tok = transformers.AutoTokenizer.from_pretrained(args.tok)
print("arch", type(hf).__name__, "layers", cfg.num_hidden_layers, "d", cfg.hidden_size, "device", dev, flush=True)
model = jlens.from_hf(hf, tok)
print("layout", model.layout, model, flush=True)

prompts = ["The quick brown fox jumps over the lazy dog and keeps running through the field until night falls.",
           "In 1905 Albert Einstein published four papers that changed physics, including special relativity."]
t0 = time.time()
lens = jlens.fit(model, prompts, dim_batch=32, max_seq_len=32, checkpoint_path=None)
print(f"fit ok in {time.time()-t0:.0f}s; source_layers={list(lens.source_layers)}", flush=True)
out = ROOT / "out" / "_b2_lenses" / "SMOKE_olmo_random.pt"
out.parent.mkdir(parents=True, exist_ok=True)
lens.save(str(out)); lens2 = jlens.JacobianLens.from_pretrained(str(out)); print("save/load ok", flush=True)

# curves-style stats (same algebra as run_b3_curves.analyze)
W = hf.get_output_embeddings().weight.float().cpu()
items = json.load(open(DATA))["items"]
ents = sorted({it[k] for it in items for k in ("intermediate", "swap_to", "answer", "swap_answer")})
ids = sorted({tok.encode(" " + w, add_special_tokens=False)[0] for w in ents if len(tok.encode(" " + w, add_special_tokens=False)) == 1})
layers = lens2.source_layers; mid = layers[len(layers) // 2]
Jm = lens2.jacobians[mid].float().cpu()
U = W[ids]
def pr(m):
    s = torch.linalg.svdvals(m); return float((s.square().sum() ** 2) / (s.pow(4).sum()))
dirs = torch.nn.functional.normalize(U @ Jm, dim=1)
print(f"entities={len(ids)} eff_dim_jl_centered={pr(dirs - dirs.mean(0, keepdim=True)):.1f} eff_dim_raw_centered={pr(torch.nn.functional.normalize(U,dim=1)-torch.nn.functional.normalize(U,dim=1).mean(0,keepdim=True)):.1f}", flush=True)
Uv = W[list(range(0, W.shape[0], max(1, W.shape[0] // 512)))[:512]]
drift = [float(torch.nn.functional.cosine_similarity(Uv @ lens2.jacobians[l].float().cpu(), Uv, dim=1).mean()) for l in layers]
print(f"mean_drift={sum(drift)/len(drift):+.3f}", flush=True)

# probe readout (same call as run_b3_probe_rank.analyze)
multi = [it for it in items if it["category"] == "multihop"][:2]
for it in multi:
    readout, _, _ = lens2.apply(model, it["prompt"], positions=[-1])
    a_id = tok.encode(" " + it["answer"], add_special_tokens=False)[0]
    i_id = tok.encode(" " + it["intermediate"], add_special_tokens=False)[0]
    deep = layers[-1]
    ar = int((readout[deep][0] > readout[deep][0][a_id]).sum())
    ir = min(int((readout[l][0] > readout[l][0][i_id]).sum()) for l in layers)
    print(f"{it['name']}: answer_rank@L{deep}={ar} inter_best_rank={ir} (readout keys={len(readout)})", flush=True)
print("SMOKE OK", flush=True)
