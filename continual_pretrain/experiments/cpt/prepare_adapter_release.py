"""Copy the publishable adapter files from a training checkpoint into a clean folder, with MODEL_CARD.md as README.

    python experiments/cpt/prepare_adapter_release.py <checkpoint_dir> <output_dir>
    huggingface-cli upload <repo-id> <output_dir> .
"""
import argparse
import shutil
from pathlib import Path

KEEP = ["adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json"]
MODEL_CARD = Path(__file__).parent / "MODEL_CARD.md"

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint_dir", type=Path)
ap.add_argument("output_dir", type=Path)
ap.add_argument("--model-card", type=Path, default=MODEL_CARD)
args = ap.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
for name in KEEP:
    if (args.checkpoint_dir / name).exists():
        shutil.copy2(args.checkpoint_dir / name, args.output_dir / name)
shutil.copy2(args.model_card, args.output_dir / "README.md")
print(f"-> {args.output_dir}: {sorted(p.name for p in args.output_dir.iterdir())}")
