import json
import os
import datetime
import logging
import sys
import argparse
import re
from collections import Counter

from tqdm import tqdm

# ───── LLM client imports (keep your project paths) ────────
from medask.ummon.openai import UmmonOpenAI
from medask.ummon.deepseek import UmmonDeepSeek
from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
# ───────────────────────────────────────────────────────────

logger = logging.getLogger("benchmark.triage_benchmark")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

TRIAGE_ORDER = {"sc": 1, "ne": 2, "em": 3}
TRIAGE_LEVELS = ["em", "ne", "sc"]

# ----------------------- Prompts -------------------------

ACTIVE_PROMPT = """
Please give a triage classification for the following vignette.
The criteria for triage classifications are presented below:
Emergency (em): Call 911 or go directly to the emergency room;
Non-Emergency (ne): Try to see a doctor, but the symptoms do not require immediate attention at an emergency room. For example, you can see a doctor in the next week.
Self-care (sc): Let the health issue get better on its own and review the situation in a few days again.

RESPONSE FORMAT: 
    Provide only one of the following responses:  
    - `em`  
    - `ne`  
    - `sc`

Vignette:
{vignette}
"""

# ───────────────────────── Helper ──────────────────────────

def _llm_triage(client, vignette_text: str) -> str:
    prompt = ACTIVE_PROMPT.format(vignette=vignette_text)
    raw = client.inquire(CMessage(user_id=1, body=prompt, role=Role.USER)).body
    cleaned = re.sub(r"[`\s]", " ", raw.lower()).strip()
    match = re.search(r"\b(em|ne|sc)\b", cleaned)
    return match.group(1) if match else cleaned[:50]


def evaluate_single(case_id: int,
                    desc: str,
                    gold: str,
                    run_id: int,
                    model_name: str,
                    client,
                    out_file):
    pred = _llm_triage(client, desc)
    correct = pred == gold
    rec = {
        "run_id": run_id,
        "case_id": case_id,
        "true_urgency": gold,
        "llm_output": pred,
        "correct": correct,
        "model": model_name,
    }
    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

# ───────────────────────── Main ───────────────────────────


def main():
    parser = argparse.ArgumentParser("LLM triage benchmark – paired logging, fixed counters")
    parser.add_argument("--model", choices=["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-4o", "gpt-4.5-preview", "deepseek-chat", "deepseek-reasoner"],
                        default="deepseek-chat")
    parser.add_argument("--vignette_set", choices=["semigran", "kopka"], default="semigran")
    parser.add_argument("--runs", type=int, default=1, help="How many stochastic passes per vignette")
    args = parser.parse_args()

    # Client factory
    if args.model in {"o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-4o", "gpt-4.5-preview"}:
        client = UmmonOpenAI(args.model)
    else:
        client = UmmonDeepSeek(args.model)

    vignette_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vignettes",
                               f"{args.vignette_set}_vignettes.jsonl")
    if not os.path.exists(vignette_fp):
        logger.error("Vignette file not found: %s", vignette_fp)
        sys.exit(1)

    with open(vignette_fp, encoding="utf-8") as f:
        vignettes = [json.loads(l) for l in f]
    num_cases = len(vignettes)
    logger.info("Loaded %d vignettes", num_cases)

    # Output path
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, f"{ts}_{args.model}_{args.vignette_set}_triage.jsonl")
    logger.info("Writing JSONL to %s", out_fp)

    # ---------------- Counters (all predictions) ----------------
    total_pred_counter = Counter()      # correct / incorrect
    per_level_total = Counter()         # keyed by triage level – counts **predictions**
    per_level_correct = Counter()
    safe_predictions = 0
    overtriage_errors = 0

    with open(out_fp, "w", encoding="utf-8") as f_out:
        for run in range(1, args.runs + 1):
            logger.info("Run %d/%d", run, args.runs)
            for idx, v in tqdm(list(enumerate(vignettes, 1)), desc=f"Run {run}"):
                rec = evaluate_single(idx,
                                      v["case_description"],
                                      v["urgency_level"].strip().lower(),
                                      run, args.model, client, f_out)

                gold = rec["true_urgency"]
                pred = rec["llm_output"]

                # Per‑prediction counts (fixed)
                per_level_total[gold] += 1
                if rec["correct"]:
                    per_level_correct[gold] += 1
                    total_pred_counter["correct"] += 1
                else:
                    total_pred_counter["incorrect"] += 1

                # Safety / over‑triage on each prediction
                if pred in TRIAGE_ORDER and gold in TRIAGE_ORDER:
                    if TRIAGE_ORDER[pred] >= TRIAGE_ORDER[gold]:
                        safe_predictions += 1
                    if (not rec["correct"]) and TRIAGE_ORDER[pred] > TRIAGE_ORDER[gold]:
                        overtriage_errors += 1

    # ---------------- Report ----------------
    total_preds = sum(total_pred_counter.values())
    overall_acc = total_pred_counter["correct"] / total_preds if total_preds else 0.0

    print("\nTriage Evaluation Summary (pooled across runs):")
    print(f"Total model calls: {total_preds}")
    print(f"Overall Accuracy: {overall_acc:.2%}\t({total_pred_counter['correct']} / {total_preds})\n")

    print("Accuracy by Triage Level (all predictions):")
    for lvl in TRIAGE_LEVELS:
        preds_lvl = per_level_total[lvl]
        corr_lvl = per_level_correct[lvl]
        acc_lvl = corr_lvl / preds_lvl if preds_lvl else 0.0
        print(f"  {lvl}: {acc_lvl:.2%}\t({corr_lvl}/{preds_lvl})")

    safety_rate = safe_predictions / total_preds if total_preds else 0.0
    incorrect_preds = total_pred_counter["incorrect"]
    overtriage_rate = overtriage_errors / incorrect_preds if incorrect_preds else 0.0

    print(f"\nSafety (at‑or‑above correct urgency): {safety_rate:.2%}\t({safe_predictions}/{total_preds})")
    print(f"Inclination to Over‑triage (among incorrect): {overtriage_rate:.2%}\t({overtriage_errors}/{incorrect_preds})")


if __name__ == "__main__":
    main()
