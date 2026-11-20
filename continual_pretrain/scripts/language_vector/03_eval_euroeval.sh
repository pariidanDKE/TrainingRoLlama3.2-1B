#!/bin/bash
# EuroEval (test split, few-shot, chat template) on Instruct and Instruct + vector, Romanian suite and the
# English counterparts. EuroEval pins its own vLLM, so it lives in a separate venv:
#   uv venv ~/.venvs/euroeval && uv pip install "euroeval[generative]==18.4.0"
# EuroEval skips datasets already in a model's results file, so re-running resumes.
set -u
umask 022   # newer NLTK refuses group-writable data dirs, which breaks IFEval scoring
cd "$(dirname "$0")/../.."
EUROEVAL="${EUROEVAL:-$HOME/.venvs/euroeval/bin/euroeval}"
declare -A MODEL_PATH=([instruct]=unsloth/Llama-3.2-1B-Instruct [vector]=$PWD/outputs/merged/rollama_chat_vector/lambda_0.5_novocab)
EN_DATASETS="sst5 conll-en squad cnn-dailymail ifeval ragtruth-en"

run() {  # run <out dir> <model> <dataset selection...>
  mkdir -p "$1"
  (cd "$1" && "$EUROEVAL" --model "${MODEL_PATH[$2]}" "${@:3}" --evaluate-test-split --few-shot \
     --generative-type instruction_tuned --gpu-memory-utilization 0.85 --no-progress-bar) > "$1/run.log" 2>&1 \
    || echo "[FAIL] $1 (see $1/run.log)"
}
for m in instruct vector; do
  run eval_results/euroeval/$m $m --language ro
  run eval_results/euroeval_en/$m $m $(for d in $EN_DATASETS; do echo --dataset $d; done)
done
