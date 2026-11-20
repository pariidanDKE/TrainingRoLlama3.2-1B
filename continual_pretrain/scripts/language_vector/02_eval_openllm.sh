#!/bin/bash
# OpenLLM-Ro benchmarks (0-shot, plain text, bf16) for every chat-side model.
# Each (model, task) is its own lm_eval call; re-running skips finished ones.
#   MODELS="instruct lambda_0.5_novocab" bash scripts/language_vector/02_eval_openllm.sh
set -u
cd "$(dirname "$0")/../.."
OUT=eval_results/chat_vector_merge/bench
M=outputs/merged/rollama_chat_vector

declare -A MODEL_PATH=(
  [base]=unsloth/Llama-3.2-1B
  [rollama]=outputs/merged/cpt_bf16/lambda_1
  [instruct]=unsloth/Llama-3.2-1B-Instruct
  [lambda_0.25]=$M/lambda_0.25 [lambda_0.5]=$M/lambda_0.5 [lambda_0.75]=$M/lambda_0.75 [lambda_1]=$M/lambda_1
  [lambda_0.5_novocab]=$M/lambda_0.5_novocab
  [lambda_0.5_tied-lm_head]=$M/lambda_0.5_tied-lm_head
)
MODELS="${MODELS:-${!MODEL_PATH[*]}}"
TASKS="_ro_arc_challenge _ro_hellaswag _ro_mmlu _ro_winogrande _ro_truthfulqa ro_wiki wikitext"

for model in $MODELS; do
  for task in $TASKS; do
    odir="$OUT/$model/$task/fs0"
    if ls "$odir"/*/results_*.json >/dev/null 2>&1; then echo "[skip] $model $task"; continue; fi
    echo "=== $model :: $task ==="
    lm_eval --model hf --model_args "pretrained=${MODEL_PATH[$model]},dtype=bfloat16,max_length=2048" \
      --tasks "$task" --num_fewshot 0 --device cuda:0 --batch_size 16 --output_path "$odir" \
      || echo "[FAIL] $model $task"
  done
done
