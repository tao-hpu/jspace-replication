"""E7 consumption deadline: at which layer does the answer commit, per task?

The three-domain zone maps (2026-07-16) show the late edge of the writable
zone moving with the task (states earliest, currency latest, in every model
with a usable tail). The two-force reading says that edge is the depth where
this task's answer-retrieval commits. This measures that depth independently
of any intervention, with a plain logit lens on clean prompts:

    for each layer l, project the last-position hidden state through the
    model's final norm and unembedding; the commit layer is the earliest l
    from which the correct answer's first token is argmax at every layer
    l' >= l (stable commit, no interventions anywhere).

If the deadline model is right, mean commit depth per domain should
reproduce the zone-map tail ordering model by model.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python \
        experiments/e7-perspectival-capture/run_e7_deadline.py [1.7b|4b|...]
      (loops all three domains in one model load)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family  # noqa: E402

from run_e7 import DOMAINS  # noqa: E402
from run_e7_healing import CaptureHooks  # noqa: E402


def main(model_key: str = "1.7b") -> None:
    model_id, lens_file, _ = MODELS[model_key]
    hub_id = model_id
    model_id = os.environ.get("JSPACE_MODEL_DIR", model_id)
    device_map = os.environ.get("JSPACE_DEVICE_MAP")
    if device_map:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device_map).eval()
        device = hf.device
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    W = hf.get_output_embeddings().weight
    final_norm = hf.model.norm
    n_layers = model.n_layers

    out = ROOT / "results" / f"e7_deadline_{family(hub_id)}{model_key.replace('.', '')}.json"
    domains_out = {}

    n_steps = int(os.environ.get("E7_DEADLINE_STEPS", "12"))

    with torch.no_grad():
        for domain, (facts, template, _tag) in DOMAINS.items():
            commits, skipped = [], 0
            for a, cap_a in facts:
                t_ans = tok.encode(" " + cap_a, add_special_tokens=False)[0]
                ids = tok(template.format(a=a), return_tensors="pt").input_ids.to(device)
                # greedy-decode until the answer token is about to be emitted;
                # the deadline is measured at THAT position (QA templates make
                # the model open with filler like " The capital of ... is").
                past, cur, hit = None, ids, None
                for _step in range(n_steps):
                    with CaptureHooks(model.layers) as cap:
                        fwd = hf(cur, past_key_values=past, use_cache=True)
                    past = fwd.past_key_values
                    nxt = int(fwd.logits[0, -1].argmax())
                    if nxt == t_ans:
                        hit = cap.states  # layer -> [seq, d] of this forward
                        break
                    cur = torch.tensor([[nxt]], device=ids.device)
                if hit is None:
                    skipped += 1  # answer token never argmax within the window
                    continue
                arg = []
                norm_p = next(final_norm.parameters())
                for l in range(n_layers):
                    # under device_map=auto, final_norm and W may sit on
                    # different cards; norm on its own device, matmul on W's
                    h = hit[l][-1].to(norm_p.device, norm_p.dtype)
                    normed = final_norm(h).float().to(W.device)
                    arg.append(int((normed @ W.float().T).argmax()))
                commit = n_layers  # never stably committed before the head
                for l in range(n_layers - 1, -1, -1):
                    if arg[l] != t_ans:
                        commit = l + 1
                        break
                    commit = l
                commits.append(commit)
            mean_commit = sum(commits) / len(commits) if commits else None
            domains_out[domain] = {
                "n_used": len(commits), "n_skipped": skipped,
                "commit_layers": commits,
                "mean_commit": mean_commit,
                "mean_commit_frac": (mean_commit / n_layers) if commits else None,
            }
            print(f"{domain:>9s}: n={len(commits)} (skip {skipped}) "
                  f"mean commit L{mean_commit:.1f} "
                  f"({mean_commit / n_layers:.2f} frac)" if commits else
                  f"{domain:>9s}: no usable items")

    out.write_text(json.dumps({
        "model": hub_id, "n_layers": n_layers,
        "band_start": round(BAND_START_FRAC * n_layers),
        "domains": domains_out,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1.7b")
