#!/bin/bash
# Add the Romanian language vector to each model. Outputs go to outputs/merged/.
set -e
cd "$(dirname "$0")/../.."
merge() { python experiments/language_vector/merge.py "$@"; }

INSTRUCT=unsloth/Llama-3.2-1B-Instruct
EMBED=nvidia/llama-3.2-nv-embedqa-1b-v2
RERANK=nvidia/llama-3.2-nv-rerankqa-1b-v2

# RoLlama-3.2-1B in bf16 (base + full vector), the "Base + CPT" reference
merge --target unsloth/Llama-3.2-1B --lam 1 --merge-special-rows --out outputs/merged/cpt_bf16/lambda_1

# chat model: λ sweep with the full vector, then the two λ = 0.5 variants
for lam in 0.25 0.5 0.75 1; do
  merge --target $INSTRUCT --lam $lam --out outputs/merged/rollama_chat_vector/lambda_$lam
done
merge --target $INSTRUCT --lam 0.5 --vocab none    --out outputs/merged/rollama_chat_vector/lambda_0.5_novocab
merge --target $INSTRUCT --lam 0.5 --vocab lm_head --out outputs/merged/rollama_chat_vector/lambda_0.5_tied-lm_head

# embedding model and reranker: no lm_head, so the vocabulary delta goes into the shared embedding matrix
for model in embed rerank; do
  if [ $model = embed ]; then target=$EMBED; else target=$RERANK; fi
  for lam in 0.25 0.5 1; do
    merge --target $target --lam $lam --vocab embed --out outputs/merged/nv_${model}_ro/lambda_$lam
  done
  merge --target $target --lam 0.5 --vocab none    --out outputs/merged/nv_${model}_ro/lambda_0.5_novocab
  merge --target $target --lam 0.5 --vocab lm_head --out outputs/merged/nv_${model}_ro/lambda_0.5_lm_head
done
