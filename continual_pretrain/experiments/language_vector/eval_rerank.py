"""Evaluate the NVIDIA reranker (or a merge of it) on MTEB, retrieve-then-rerank.

1. first-stage (once per suite): the embedding model retrieves the top 20 candidates per query; saved so
   every reranker sees the same candidates. WikipediaRerankingMultilingual ships its own candidates.
2. rerank: the reranker rescores them, with the prompt "question:{q} \\n \\n passage:{p}" from its model card.

    python experiments/language_vector/eval_rerank.py first-stage
    python experiments/language_vector/eval_rerank.py rerank --model nvidia/llama-3.2-nv-rerankqa-1b-v2 --name nv-rerank
    python experiments/language_vector/eval_rerank.py --suite en_paired rerank --model <merged dir> --name <name>
"""
import argparse
import json
from pathlib import Path

import mteb
import numpy as np
import torch
from mteb.models.model_meta import ModelMeta
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path("eval_results/rerank_merge")
# BelebeleRetrieval is only used for the first stage: mteb can't convert it to a reranking task.
# WebFAQ's English corpus (5.3M documents) is too large to embed on one GPU, so en_paired leaves it out.
SUITES = {
    "ro": dict(lang="ron", out=ROOT, retrieval=["BelebeleRetrieval", "WikipediaRetrievalMultilingual", "XQuADRetrieval", "WebFAQRetrieval"],
               rerankable=["WikipediaRetrievalMultilingual", "XQuADRetrieval", "WebFAQRetrieval"]),
    "en_paired": dict(lang="eng", out=ROOT / "en_paired", retrieval=["BelebeleRetrieval", "WikipediaRetrievalMultilingual", "XQuADRetrieval"],
                      rerankable=["WikipediaRetrievalMultilingual", "XQuADRetrieval"]),
}
RERANK_TASKS = ["WikipediaRerankingMultilingual"]


def get_tasks(names, lang):
    return list(mteb.get_tasks(tasks=names, languages=[lang], exclusive_language_filter=True))


class NvReranker:
    def __init__(self, model_name, batch_size=32, max_length=512):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()
        self.model.config.pad_token_id = self.model.config.pad_token_id or self.tok.eos_token_id
        self.batch_size, self.max_length = batch_size, max_length
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={"name": f"local/{Path(model_name).name}", "revision": "local"})

    @torch.inference_mode()
    def predict(self, inputs1, inputs2, **kwargs):
        texts = [f"question:{q} \n \n passage:{p}" for qb, pb in zip(inputs1, inputs2) for q, p in zip(qb["text"], pb["text"])]
        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to("cuda")
            scores.append(self.model(**batch).logits.view(-1).float().cpu().numpy())
        return np.concatenate(scores)


def mean_scores(res, keys=("main_score",)):
    out = {}
    for tr in res.task_results:
        s = [x for split in tr.scores.values() for x in split]
        out[tr.task_name] = {k: float(np.mean([x[k] for x in s])) for k in keys}
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--suite", choices=list(SUITES), default="ro")
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("first-stage").add_argument("--retriever", default="nvidia/llama-3.2-nv-embedqa-1b-v2")
rr = sub.add_parser("rerank")
rr.add_argument("--model", required=True)
rr.add_argument("--name", required=True)
args = ap.parse_args()
S = SUITES[args.suite]
first_stage_dir = S["out"] / "_first_stage"

if args.cmd == "first-stage":
    from sentence_transformers import SentenceTransformer
    retriever = SentenceTransformer(args.retriever, trust_remote_code=True, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    retriever.max_seq_length = 512
    first_stage_dir.mkdir(parents=True, exist_ok=True)
    res = mteb.evaluate(retriever, tasks=get_tasks(S["retrieval"], S["lang"]), encode_kwargs={"batch_size": 64},
                        cache=mteb.ResultCache(cache_path=str(first_stage_dir / "cache")),
                        overwrite_strategy="always", prediction_folder=first_stage_dir)
    summary = {"retriever": args.retriever, "scores": mean_scores(res)}
    (first_stage_dir / "summary.json").write_text(json.dumps(summary, indent=2))
else:
    tasks = get_tasks(RERANK_TASKS, S["lang"]) + [t.convert_to_reranking(first_stage_dir, top_k=20)
                                                  for t in get_tasks(S["rerankable"], S["lang"])]
    out = S["out"] / args.name
    out.mkdir(parents=True, exist_ok=True)
    res = mteb.evaluate(NvReranker(args.model), tasks=tasks, encode_kwargs={"batch_size": 32},
                        cache=mteb.ResultCache(cache_path=str(out / "cache")), overwrite_strategy="only-missing")
    summary = {"model": args.model, "scores": mean_scores(res, ("main_score", "ndcg_at_10", "mrr_at_10"))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["scores"], indent=2))
