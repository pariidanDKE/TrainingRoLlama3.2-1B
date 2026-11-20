"""Print EuroEval results as a markdown table: dataset x model, main metric ± 95% CI half-width.

    python experiments/language_vector/euroeval_table.py [models...] [--root eval_results/euroeval_en]
"""
import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, tuple[str, float, float]]:
    """dataset -> (metric, score, CI half-width) for the dataset's first (main) metric."""
    out = {}
    for line in path.open():
        for r in json.loads(line)["evaluation_results"]:
            ci = r["score_details"].get("uncertainty", {}).get("confidence_interval")
            out.setdefault(r["source_data"]["dataset_name"], (
                r["evaluation_name"].removeprefix("test_"), r["score_details"]["score"],
                (ci["upper"] - ci["lower"]) / 2 if ci else 0.0))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["instruct", "vector"])
    ap.add_argument("--root", type=Path, default=Path("eval_results/euroeval"))
    args = ap.parse_args()
    res = {m: load(args.root / m / "euroeval_benchmark_results.jsonl") for m in args.models}
    print("| dataset | metric | " + " | ".join(args.models) + " |\n|---|---|" + "---|" * len(args.models))
    for ds in dict.fromkeys(d for r in res.values() for d in r):
        metric = next(r[ds][0] for r in res.values() if ds in r)
        cells = [f"{res[m][ds][1]:.1f} ± {res[m][ds][2]:.1f}" if ds in res[m] else "–" for m in args.models]
        print(f"| {ds} | {metric} | " + " | ".join(cells) + " |")
