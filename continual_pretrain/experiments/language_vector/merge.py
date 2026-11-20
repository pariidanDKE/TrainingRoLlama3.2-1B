"""Add the Romanian language vector to a model built on Llama-3.2-1B:  W = W_target + λ · ΔW.

The vocabulary part of the vector (embed_tokens / lm_head deltas) is handled by --vocab:
  untied   embed_tokens and lm_head each get their own delta; lm_head becomes a separate matrix (chat model)
  embed / lm_head / mean   one shared matrix gets the embedding delta, the lm_head delta, or their mean
  none     transformer layers only (the 90 MB version used in the article)
Rows of special tokens (id >= 128000) keep the target's weights unless --merge-special-rows is set.

    python experiments/language_vector/merge.py --target unsloth/Llama-3.2-1B-Instruct --lam 0.5 --vocab none \
        --out outputs/merged/rollama_chat_vector/lambda_0.5_novocab
"""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from deltas import ADAPTER, VOCAB, compute_deltas, match_key

SPECIAL = 128000

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--lam", type=float, required=True)
ap.add_argument("--vocab", choices=["untied", "embed", "lm_head", "mean", "none"], default="untied")
ap.add_argument("--merge-special-rows", action="store_true")
ap.add_argument("--adapter", default=ADAPTER)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

target_dir = Path(args.target) if Path(args.target).is_dir() else Path(snapshot_download(args.target, ignore_patterns=["*.bin", "*.pt", "original/*"]))
weights_file = next(target_dir.glob("*.safetensors"))
weights = load_file(weights_file)

deltas = compute_deltas(args.adapter)
embed, head = (deltas.pop(k) for k in VOCAB)
if args.vocab == "untied":
    deltas.update(zip(VOCAB, (embed, head)))
    weights.setdefault("lm_head.weight", weights["model.embed_tokens.weight"].clone())
elif args.vocab != "none":
    deltas[VOCAB[0]] = {"embed": embed, "lm_head": head, "mean": (embed + head) / 2}[args.vocab]

for name, d in deltas.items():
    key = match_key(name, weights)
    d = d * args.lam
    if name in VOCAB and not args.merge_special_rows:
        d[SPECIAL:] = 0
    weights[key] = (weights[key].float() + d).to(weights[key].dtype)

# copy the rest of the target repo (config, tokenizer, chat template, custom modelling code), then the new weights
shutil.copytree(target_dir, args.out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pt", ".git*"))
save_file({k: v.contiguous() for k, v in weights.items()}, args.out / weights_file.name, metadata={"format": "pt"})
if "lm_head.weight" in weights:
    cfg = json.loads((args.out / "config.json").read_text())
    (args.out / "config.json").write_text(json.dumps({**cfg, "tie_word_embeddings": False}, indent=2))
(args.out / "merge_info.json").write_text(json.dumps(vars(args), default=str, indent=2))
print(f"-> {args.out}")
