#!/bin/bash
# Full OpenLLM-Ro suite eval, bf16, matching their reported MULTI-FEW-SHOT protocol.
#
# WHY PER (task, few-shot) CALLS:
#   - The fork's GROUP tasks (ro_arc_challenge, ...) are broken in this lm-eval version
#     (sub-entries lack unique names). So we drive each task at each few-shot level with
#     an explicit --num_fewshot, then AVERAGE across few-shot per task (= their protocol).
#   - Each (task, few-shot) is a SEPARATE lm_eval call -> a results file after every cell,
#     so a late failure (OOM / bad metric) costs only that cell. Re-running RESUMES
#     (completed cells are skipped).
#   - bf16, NOT 4-bit: clean, comparable, full-precision numbers.
#
# Two passes: base Llama-3.2-1B (baseline row) + base+adapter (RoLlama row).
#
# Usage:
#   bash scripts/run_openllm_ro_suite.sh            # FULL run (no doc limit)
#   LIMIT=100 bash scripts/run_openllm_ro_suite.sh  # smoke (100 docs per cell)

set -u
cd "$(dirname "$0")/../.."   # run from continual_pretrain/

# --- Config -----------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-unsloth/Llama-3.2-1B}"         # env-overridable; 4-bit: unsloth/llama-3.2-1b-unsloth-bnb-4bit
ADAPTER="${ADAPTER:-outputs/release/rollama-3.2-1b-ro-cpt-filtered}" # env-overridable
DTYPE="bfloat16"
DEVICE="cuda:0"
BATCH_SIZE="${BATCH_SIZE:-8}"        # env-overridable; 4-bit 1B has lots of VRAM headroom
LIMIT="${LIMIT-}"                                       # empty = full run; e.g. LIMIT=100 for smoke
OUT_ROOT="${OUT_ROOT:-eval_results/openllm_ro_suite_bf16}"   # override: OUT_ROOT=... bash ...

# Per-task few-shot lists = OpenLLM-Ro canonical settings (averaged per task).
# NOTE: laroseda is OMITTED -- its dataset loads via a script (laroseda.py), which
# datasets>=3.0 rejects. Needs a parquet source / pinned datasets; see eval_alignment/SPEC.md.
declare -A FEWSHOT=(
  [_ro_arc_challenge]="0 1 3 5 10 25"
  [_ro_mmlu]="0 1 3 5"
  [_ro_winogrande]="0 1 3 5"
  [_ro_hellaswag]="0 1 3 5 10"
  [_ro_gsm8k]="1 3 5"
  [_ro_truthfulqa]="0"
  [_ro_xquad]="0 1 3 5"
  [_ro_sts]="1 3 5"
  [wmt-en-ro]="0 1 3 5"
  [wmt-ro-en]="0 1 3 5"
  [ro_wiki]="0"          # perplexity -- few-shot irrelevant, run once
  # English forgetting check (not in OpenLLM-Ro table):
  [arc_challenge]="25"
  [winogrande]="5"
  [wikitext]="0"
)

LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG="--limit $LIMIT"
    echo ">>> SMOKE: --limit $LIMIT per (task, few-shot) cell"
else
    echo ">>> FULL RUN: no doc limit (this is long -- generative tasks dominate)"
fi

run_pass() {
    local label=$1 model_args=$2 out=$3
    echo ""
    echo "########################################"
    echo "### PASS: $label"
    echo "########################################"
    # ONLY_TASKS (space-separated keys) restricts the run to a subset; empty = all.
    local tasks="${ONLY_TASKS:-${!FEWSHOT[*]}}"
    for task in $tasks; do
        for k in ${FEWSHOT[$task]}; do
            local odir="${out}/${task}/fs${k}"
            if ls "${odir}"/*/results_*.json >/dev/null 2>&1; then
                echo "[skip] $label  $task  ${k}-shot  (already done)"
                continue
            fi
            echo ""
            echo "=== $label :: $task :: ${k}-shot ==="
            # base model: no chat template (likelihood / perplexity / generation tasks).
            lm_eval --model hf \
                --model_args "$model_args" \
                --tasks "$task" \
                --num_fewshot "$k" \
                --device "$DEVICE" \
                --batch_size "$BATCH_SIZE" \
                $LIMIT_FLAG \
                --output_path "$odir" \
                || echo "[FAIL] $label  $task  ${k}-shot  (continuing)"
        done
    done
}

run_pass "BASE: ${BASE_MODEL}" \
    "pretrained=${BASE_MODEL},dtype=${DTYPE}" \
    "${OUT_ROOT}/base"

run_pass "ADAPTER: ${ADAPTER}" \
    "pretrained=${BASE_MODEL},peft=${ADAPTER},dtype=${DTYPE}" \
    "${OUT_ROOT}/rollama"

echo ""
echo "Done. Per-cell results under ${OUT_ROOT}/{base,rollama}/<task>/fs<k>/"
echo "Next: aggregate per-task averages across few-shot for the model card."
