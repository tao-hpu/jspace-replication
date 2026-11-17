#!/bin/bash
# One-shot OLMo-1B-0724 developmental sweep for Paper B (cross-family
# check). Run on a single GPU under nohup:
#   nohup bash experiments/e11-development/olmo_4090.sh > scratchpad/olmo_4090.log 2>&1 &
# Stages: (1) resumable curl download of each checkpoint into $B_LOCAL_ROOT/step{N}/,
# (2) lens fit (paper recipe: 128 WikiText prompts, offset 0, 128 tokens), (3) curves,
# (4) probe rank with 5-rotation foil null, (5) induction heads. Checkpoints are processed
# in PRIORITY order so a partial run already yields a usable table.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/jspace-venv/bin/python}
export B_LOCAL_ROOT=${B_LOCAL_ROOT:-$HOME/ckpts/olmo-1b-0724}
MODEL=allenai/OLMo-1B-0724-hf
HF=${HF_BASE:-https://huggingface.co}
# step -> hub branch suffix (tokens), from the hub refs listing 2026-09-24
declare -A TOK=( [0]=tokens0B [1000]=tokens2B [2000]=tokens4B [3000]=tokens6B [4000]=tokens8B
  [6000]=tokens12B [9000]=tokens18B [13000]=tokens27B [23000]=tokens48B [43000]=tokens90B
  [73000]=tokens153B [103000]=tokens215B [143000]=tokens299B )
PRIORITY=(1000 6000 13000 43000 143000 0 2000 4000 9000 23000 73000 103000 3000)

mkdir -p "$B_LOCAL_ROOT/tokenizer" scratchpad
for f in tokenizer.json tokenizer_config.json special_tokens_map.json config.json; do
  [ -s "$B_LOCAL_ROOT/tokenizer/$f" ] || curl -sS -L --retry 10 -o "$B_LOCAL_ROOT/tokenizer/$f" "$HF/$MODEL/resolve/main/$f"
done

download() {  # $1 = step
  local n=$1 rev="step$1-${TOK[$1]}" dst="$B_LOCAL_ROOT/step$1"
  mkdir -p "$dst"
  for f in config.json generation_config.json model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
    curl -sS -L -C - --retry 30 --retry-delay 10 --retry-all-errors -o "$dst/$f" "$HF/$MODEL/resolve/$rev/$f" || return 1
  done
  # integrity: shard sizes must match the index (index lists total_size)
  $PY - "$dst" <<'PYI' || return 1
import json,os,sys
d=sys.argv[1]; idx=json.load(open(os.path.join(d,"model.safetensors.index.json")))
tot=sum(os.path.getsize(os.path.join(d,f)) for f in set(idx["weight_map"].values()))
exp=idx["metadata"]["total_size"]
# safetensors files carry an 8-byte length + JSON header on top of tensor bytes; allow that slack
assert tot >= exp, (tot, exp); print("sizes ok", tot, exp)
PYI
}

DONE=()
for n in "${PRIORITY[@]}"; do
  echo "=== $(date '+%F %T') step$n: download"; download $n || { echo "DOWNLOAD FAILED step$n"; continue; }
  echo "=== $(date '+%F %T') step$n: fit"
  # other tenants may hold GPU memory: step down the Jacobian batch on failure (e.g. OOM)
  ok=0
  for db in ${DIM_BATCHES:-32 16 8}; do
    $PY experiments/e11-development/run_b2_fit.py --model $MODEL --revisions step$n --dim-batch $db && { ok=1; break; }
    echo "fit failed at --dim-batch $db, retrying smaller"
  done
  [ $ok = 1 ] || { echo "FIT FAILED step$n"; continue; }
  DONE+=($n)
  echo "=== $(date '+%F %T') step$n: curves + probe-rank so far (${DONE[*]})"
  $PY experiments/e11-development/run_b3_curves.py --model $MODEL
  $PY experiments/e11-development/run_b3_probe_rank.py --model $MODEL --foil 5 --revisions $(printf 'step%s ' "${DONE[@]}")
done
echo "=== $(date '+%F %T') induction"
$PY experiments/e11-development/run_b3_induction.py --model $MODEL --steps $(printf '%s ' "${DONE[@]}")
echo "=== ALL DONE $(date '+%F %T') steps: ${DONE[*]}"
