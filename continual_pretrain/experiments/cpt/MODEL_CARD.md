---
license: llama3.2
language:
- ro
base_model:
- meta-llama/Llama-3.2-1B
library_name: peft
tags:
- romanian
- continual-pretraining
- qlora
- 4-bit
- bitsandbytes
- llama
---

# Model Card for RoLlama-3.2-1B

RoLlama-3.2-1B is a continually-pretrained adaptation of `meta-llama/Llama-3.2-1B` for the Romanian language, produced under a constrained compute budget (single RTX 3090). It was trained with QLoRA on **2.4B Romanian tokens** (FineWeb2-Edu-Ro, quality-filtered) mixed with English (80/20) to limit catastrophic forgetting.

> **Designed for 4-bit (QLoRA) inference under constrained VRAM.** The adapter was trained on a 4-bit (nf4) base and is intended to run in that configuration. Reported metrics use 4-bit inference — its deployment precision.

## Model Details

### Model Description

- **Developed by:** Dan Parii
- **Language(s):** Romanian (ro), English (en)
- **License:** Llama 3.2 Community License
- **Continually pretrained from:** `meta-llama/Llama-3.2-1B`
- **Training corpus:** FineWeb2-Edu-Ro (educational quality score = 4 subset) + FineWeb (EN), 80/20 mix, 2.4B tokens
- **Method:** QLoRA (4-bit), LoRA rank 64 incl. `embed_tokens` + `lm_head`, WSD scheduler, no sequence packing

### Model Sources

- **Repository:** https://github.com/pariidanDKE/ContinualPretrainRo
- **Write-up:** https://dan1180627.substack.com/p/rollama32-1b-cpt-of-a-small-language
- **Model:** https://huggingface.co/danp27/RoLLama-3.2-1B

## Intended Use

### Intended Use Cases
Research on Romanian language adaptation and continual pretraining of small models. This is a **base** model (not instruction-tuned); it is intended for further fine-tuning (SFT/DPO) or text-completion research, not for direct chat/instruction following.

### Out-of-Scope Use
Any use that violates the Llama 3.2 license or applicable laws; use in languages other than Romanian/English; deployment as an assistant without further alignment.

## How to Get Started with the Model

This is a **LoRA adapter** meant to be loaded on the base model in **4-bit** (its training/deployment config):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = "meta-llama/Llama-3.2-1B"
adapter = "danp27/RoLLama-3.2-1B"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

inputs = tok("Capitala României este", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Academic Benchmarks

Accuracy (%), 4-bit inference, averaged across the standard few-shot settings. Higher is better. Average is the mean over all six tasks.

| Model | Average | ARC | MMLU | Winogrande | HellaSwag | GSM8k | TruthfulQA |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B (base)   | 31.25     | 29.45 | **24.50** | 51.82 | 35.75 | **1.21** | 44.76 |
| **RoLlama-3.2-1B**    | **32.55** | **31.33** | 23.59 | **54.14** | **40.21** | 0.18 | **45.82** |

<sub>ARC/HellaSwag report `acc_norm`; MMLU/Winogrande/TruthfulQA report `acc`; GSM8k reports `exact_match`. Scores from the final 2.4B-token checkpoint of the FineWeb2-Edu-Ro run, 4-bit inference (deployment precision).</sub>

## Downstream Tasks

### LaRoSeDa (sentiment) — Macro F1

| Model | Binary (Few-shot) | Binary (Finetuned) | Multiclass (Few-shot) | Multiclass (Finetuned) |
|---|---|---|---|---|
| Llama-3.2-1B (base) | 50.84 | - | **33.42** | - |
| RoLlama-3.2-1B      | **67.54** | - | 25.63 | - |

### WMT (translation) — BLEU

| Model | EN→RO (Few-shot) | EN→RO (Finetuned) | RO→EN (Few-shot) | RO→EN (Finetuned) |
|---|---|---|---|---|
| Llama-3.2-1B (base) | 6.11 | - | 15.60 | - |
| RoLlama-3.2-1B      | 2.10 | - | 2.94 | - |

### XQuAD (extractive QA)

| Model | EM (Few-shot) | F1 (Few-shot) | EM (Finetuned) | F1 (Finetuned) |
|---|---|---|---|---|
| Llama-3.2-1B (base) | 20.88 | 31.11 | - | - |
| RoLlama-3.2-1B      | 13.51 | 25.21 | - | - |

### STS (semantic textual similarity)

| Model | Spearman (Few-shot) | Pearson (Few-shot) | Spearman (Finetuned) | Pearson (Finetuned) |
|---|---|---|---|---|
| Llama-3.2-1B (base) | 0.019 | 0.018 | - | - |
| RoLlama-3.2-1B      | -0.004 | -0.005 | - | - |

## Additional Romanian Signals (supplementary, not in standard suite)

These are extra measurements taken during development; included for completeness.

| Metric | Llama-3.2-1B (base) | RoLlama-3.2-1B | Note |
|---|---|---|---|
| RoWiki perplexity ↓   | 60.44 | **32.47** | primary Romanian fluency signal |
| RO Belebele (acc_norm)| 26.47 | 27.22 | reading comprehension |
| RO Grammar (acc)      | 28.37 | 27.74 | |

## English Retention (catastrophic forgetting check)

| Metric | Llama-3.2-1B (base) | RoLlama-3.2-1B |
|---|---|---|
| WikiText perplexity ↓     | 12.35 | 14.87 |
| ARC-Challenge (acc_norm)  | 34.64 | 33.70 |
| Winogrande (acc)          | 61.25 | 59.67 |

<sub>English degrades only modestly, confirming the 80/20 RO/EN mix preserved English capability.</sub>

## Training Recipe (summary)

| Hyperparameter | Value |
|---|---|
| LoRA rank / alpha | 64 / 64 |
| Target modules | attn + MLP + `embed_tokens` + `lm_head` |
| Learning rate / embedding LR | 1e-4 / 2e-5 |
| Effective batch size | 128 (BS 4 × GA 32) |
| Data mix | 80% RO (FineWeb2-Edu-Ro) / 20% EN |
| Sequence packing | disabled |
| Scheduler | warmup_stable_decay (6% warmup, 5% decay) |
| Precision | BF16 + QLoRA 4-bit + FlashAttention 2 + grad checkpointing |
| Tokens | 2.4B (8 milestones × 300M) |

## Citation

```bibtex
@misc{parii2026rollama32,
  title  = {RoLlama-3.2-1B: Continual Pretraining of a Small Language Model for Romanian under Compute Constraints},
  author = {Parii, Dan},
  year   = {2026},
  howpublished = {\url{https://dan1180627.substack.com/p/rollama32-1b-cpt-of-a-small-language}}
}
```
