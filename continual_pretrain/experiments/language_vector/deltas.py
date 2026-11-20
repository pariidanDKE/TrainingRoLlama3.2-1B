"""The Romanian language vector: the weight change ΔW that the CPT LoRA adapter makes to Llama-3.2-1B.

- LoRA layers: ΔW = B @ A * scale (α/√r with rsLoRA, else α/r).
- embed_tokens / lm_head (trained in full): ΔW = trained - base. The base model ties lm_head to embed_tokens.
"""
import json
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file

ADAPTER = "OpenLLM-Ro/RoLlama-3.2-1B"  # the released CPT adapter
BASE = "unsloth/Llama-3.2-1B"
VOCAB = ("model.embed_tokens.weight", "lm_head.weight")
PREFIX = "base_model.model."


def resolve(model_id: str) -> Path:
    """Local directory for a local path or a Hugging Face hub id."""
    p = Path(model_id)
    return p if p.is_dir() else Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.py"]))


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    return {k: v for f in sorted(model_dir.glob("*.safetensors")) for k, v in load_file(f).items()}


def compute_deltas(adapter_dir=ADAPTER, base: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """{state_dict key: ΔW in fp32} for every weight the adapter changed."""
    adapter_dir = resolve(adapter_dir)
    base = base if base is not None else load_state_dict(resolve(BASE))
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
    deltas = {}
    with safe_open(adapter_dir / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            if k.endswith(".lora_A.weight"):
                name = k[len(PREFIX):].replace(".lora_A", "")
                A, B = f.get_tensor(k).float(), f.get_tensor(k.replace("lora_A", "lora_B")).float()
                deltas[name] = B @ A * scale
            elif ".lora_" not in k:  # embed_tokens / lm_head, saved as full trained copies
                name = k[len(PREFIX):].replace(".modules_to_save.default", "")
                base_w = base.get(name, base["model.embed_tokens.weight"])  # lm_head is tied in the base
                deltas[name] = f.get_tensor(k).float() - base_w.float()
    return deltas


def match_key(name: str, keys) -> str | None:
    """Encoder checkpoints often drop the "model." prefix; find the key the target model uses."""
    return next((c for c in (name, name.removeprefix("model."), "model." + name) if c in keys), None)
