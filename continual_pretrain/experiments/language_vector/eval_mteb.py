"""Evaluate an embedding model on MTEB's Romanian tasks, or on the English subsets of the same datasets.

    python experiments/language_vector/eval_mteb.py --model nvidia/llama-3.2-nv-embedqa-1b-v2 --name nv-embed
    python experiments/language_vector/eval_mteb.py --model <merged dir> --name nv-embed_l0.5_novocab --suite en_paired

Results: eval_results/embed_merge/[en_paired/]<name>/summary.json (main score per task, averaged over subsets).
"""
import argparse
import json
from pathlib import Path

import mteb
import torch
from sentence_transformers import SentenceTransformer

RO_TASKS = ["RonSTS", "Moroco.v2", "RomanianReviewsSentiment.v2", "RomanianSentimentClassification.v2",
            "MassiveIntentClassification", "SIB200Classification.v3", "SIB200ClusteringS2S",
            "WikipediaRerankingMultilingual", "BelebeleRetrieval", "XQuADRetrieval", "WikipediaRetrievalMultilingual"]
BITEXT_TASKS = ["FloresBitextMining", "Tatoeba"]  # ron <-> eng pairs
# multilingual datasets from RO_TASKS that also have an English subset
EN_PAIRED_TASKS = ["MassiveIntentClassification", "SIB200Classification.v3", "SIB200ClusteringS2S",
                   "WikipediaRerankingMultilingual", "BelebeleRetrieval", "XQuADRetrieval", "WikipediaRetrievalMultilingual"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--suite", choices=["ro", "en_paired"], default="ro")
args = ap.parse_args()

model = SentenceTransformer(args.model, trust_remote_code=True, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
model.max_seq_length = 512

if args.suite == "ro":
    tasks = [*mteb.get_tasks(tasks=RO_TASKS, languages=["ron"], exclusive_language_filter=True),
             *mteb.get_tasks(tasks=BITEXT_TASKS, languages=["ron", "eng"], exclusive_language_filter=True)]
    out = Path("eval_results/embed_merge") / args.name
else:
    tasks = mteb.get_tasks(tasks=EN_PAIRED_TASKS, languages=["eng"], exclusive_language_filter=True)
    out = Path("eval_results/embed_merge/en_paired") / args.name

res = mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 64},
                    cache=mteb.ResultCache(cache_path=str(out / "cache")), overwrite_strategy="only-missing")
scores = {}
for tr in res.task_results:
    s = [x["main_score"] for split in tr.scores.values() for x in split]
    scores[tr.task_name] = sum(s) / len(s)
    print(f"{tr.task_name:40s} {scores[tr.task_name]:.4f}")
(out / "summary.json").write_text(json.dumps({"model": args.model, "scores": scores}, indent=2))
