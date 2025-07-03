# paired_analysis.py – Paired statistical comparison of two triage-result files
# ─────────────────────────────────────────────────────────────────────────────
# Each input is a JSON‑Lines file written by triage_benchmark_paired.py and
# therefore contains one row per vignette × run × model with fields
#   run_id, case_id, true_urgency, llm_output, correct, model
#
# The script:
#   • prints overall accuracy, per‑triage‑level accuracy, safety and over‑triage
#     rates for each file
#   • performs an exact / χ² McNemar test on vignette‑paired correctness
#   • outputs odds ratio and accuracy difference (B − A)
#
# Usage (two files):
#   python paired_analysis.py results/runA.jsonl results/runB.jsonl
#
# If you pooled multiple vignette sets into each file first (e.g. cat >deepseek.jsonl),
# the pairing still works as long as run_id & case_id stay unique per vignette.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

TRIAGE_LEVELS = ["em", "ne", "sc"]
ORDER = {"sc": 1, "ne": 2, "em": 3}


# ────────────────────────── helpers ──────────────────────────

def load_jsonl(fp: Path) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame; minimal validation."""
    with fp.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if not data:
        raise ValueError(f"{fp} is empty")
    return pd.DataFrame(data)


# Removed unused accuracy function


def safety_overtriage(df: pd.DataFrame) -> Tuple[int, int]:
    """Count safe predictions and over‑triage errors in DataFrame."""
    safe = ((df.true_urgency.map(ORDER) <= df.llm_output.map(ORDER))).sum()
    over = ((df.true_urgency.map(ORDER) < df.llm_output.map(ORDER)) & (~df.correct)).sum()
    return safe, over


def print_accuracy_block(df: pd.DataFrame, label: str):
    """Pretty accuracy / safety block for one model."""
    total = len(df)
    correct_total = df.correct.sum()
    print(f"\n=== {label} ===")
    print(f"Overall accuracy: {correct_total/total:.2%}  ({correct_total}/{total})")
    print("Accuracy by triage level:")
    for lvl in TRIAGE_LEVELS:
        mask = df.true_urgency == lvl
        corr = df.loc[mask, "correct"].sum()
        n = mask.sum()
        acc = corr / n if n else 0.0
        print(f"  {lvl}: {acc:.2%}  ({corr}/{n})")
    safe, over = safety_overtriage(df)
    incorrect = total - correct_total
    print(f"Safety (at‑or‑above correct urgency): {safe/total:.2%}  ({safe}/{total})")
    otr = over / incorrect if incorrect else 0.0
    print(f"Over‑triage inclination (among incorrect): {otr:.2%}  ({over}/{incorrect})")


# ─────────────────────────── main ────────────────────────────

def main():
    parser = argparse.ArgumentParser("Paired comparison of two triage JSONL files")
    parser.add_argument("file_a", type=Path, help="JSONL file for model / prompt A")
    parser.add_argument("file_b", type=Path, help="JSONL file for model / prompt B")
    args = parser.parse_args()

    dfA = load_jsonl(args.file_a).rename(columns={"llm_output": "pred_A", "correct": "correct_A"})
    dfB = load_jsonl(args.file_b).rename(columns={"llm_output": "pred_B", "correct": "correct_B"})

    # Merge on run & vignette to ensure paired rows
    merged = pd.merge(dfA, dfB, on=["run_id", "case_id", "true_urgency"], how="inner")
    if merged.empty:
        raise ValueError("The two files share no common (run_id, case_id) pairs; cannot do paired test.")

    # ---------------------------------------------------------------- stats per model
    print_accuracy_block(merged.rename(columns={"pred_A": "llm_output", "correct_A": "correct"}),
                         label="MODEL A")
    print_accuracy_block(merged.rename(columns={"pred_B": "llm_output", "correct_B": "correct"}),
                         label="MODEL B")

    # ---------------------------------------------------------------- McNemar test
    b = ((merged.correct_A) & (~merged.correct_B)).sum()  # A right, B wrong
    c = ((~merged.correct_A) & (merged.correct_B)).sum()  # A wrong, B right
    discordant = b + c

    exact_flag = discordant < 25  # statsmodels advice
    res = mcnemar([[0, b], [c, 0]], exact=exact_flag, correction=not exact_flag)

    print("\n=== McNemar paired test ===")
    print(f"discordant pairs: A‑right/B‑wrong = {b}, A‑wrong/B‑right = {c}")
    print(f"p‑value: {res.pvalue:.4g}   (exact={exact_flag})")
    if c == 0:
        print("Odds ratio: ∞  (B never correct when A is wrong)")
    else:
        print(f"Odds ratio (A vs B): {b/c:.2f}")

    accA = merged.correct_A.mean()
    accB = merged.correct_B.mean()
    print(f"Accuracy difference (B − A): {(accB-accA):+.2%}")


if __name__ == "__main__":
    main()
