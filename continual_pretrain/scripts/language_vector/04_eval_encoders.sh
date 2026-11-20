#!/bin/bash
# MTEB for the embedding model and the reranker: Romanian tasks for every merge,
# English subsets of the same datasets for the original and the λ = 0.5 transformer-only merge.
set -e
cd "$(dirname "$0")/../.."
LV=experiments/language_vector
VARIANTS="l0.25:lambda_0.25 l0.5:lambda_0.5 l1:lambda_1 l0.5_novocab:lambda_0.5_novocab l0.5_lm_head:lambda_0.5_lm_head"

# embedding model
python $LV/eval_mteb.py --model nvidia/llama-3.2-nv-embedqa-1b-v2 --name nv-embed
python $LV/eval_mteb.py --model nvidia/llama-3.2-nv-embedqa-1b-v2 --name nv-embed --suite en_paired
for v in $VARIANTS; do
  python $LV/eval_mteb.py --model outputs/merged/nv_embed_ro/${v#*:} --name nv-embed_${v%%:*}
done
python $LV/eval_mteb.py --model outputs/merged/nv_embed_ro/lambda_0.5_novocab --name nv-embed_l0.5_novocab --suite en_paired

# reranker (first stage: the original embedding model retrieves the candidates)
for suite in ro en_paired; do
  python $LV/eval_rerank.py --suite $suite first-stage
  python $LV/eval_rerank.py --suite $suite rerank --model nvidia/llama-3.2-nv-rerankqa-1b-v2 --name nv-rerank
done
for v in $VARIANTS; do
  python $LV/eval_rerank.py rerank --model outputs/merged/nv_rerank_ro/${v#*:} --name nv-rerank_${v%%:*}
done
python $LV/eval_rerank.py --suite en_paired rerank --model outputs/merged/nv_rerank_ro/lambda_0.5_novocab --name nv-rerank_l0.5_novocab
