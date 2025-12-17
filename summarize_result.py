#!/usr/bin/env python
"""
Recalculate ABA evaluation summary metrics from existing result CSV files.

Reads:  *_aba_eval.csv  (produced by run_aba_eval.py)
Writes: aba_eval_summary_files_recalc.csv/.json/.txt

Per-file metrics:
- attempted_n: number of QA rows
- scored_n: number of rows with a valid integer score in [0,5]
- parse_fail_n: attempted_n - scored_n
- mean_score, prop_ge_4, prop_5
- same metrics per category (91–94)

Also computes:
- MACRO_AVG: average across files (means and proportions are averaged; counts summed)
- MICRO_AVG: pooled across all scored QA rows (exact mean/props)
"""

import argparse
import csv
import glob
import json
import os
import statistics
from typing import Dict, Any, List, Optional, Tuple

TARGET_CATEGORIES = [91, 92, 93, 94]


def parse_score(x: str) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        v = int(s)
    except Exception:
        return None
    return v if 0 <= v <= 5 else None


def safe_mean(xs: List[int]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def safe_prop(preds: List[bool]) -> Optional[float]:
    return (sum(preds) / len(preds)) if preds else None


def load_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def compute_file_metrics(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    attempted_n = len(rows)
    scores_all = []
    scored_rows = 0

    for r in rows:
        sc = parse_score(r.get("score", ""))
        if sc is not None:
            scores_all.append(sc)
            scored_rows += 1

    metrics: Dict[str, Any] = {
        "overall_attempted_n": attempted_n,
        "overall_scored_n": scored_rows,
        "overall_parse_fail_n": attempted_n - scored_rows,
        "overall_mean_score": safe_mean(scores_all),
        "overall_prop_ge_4": safe_prop([s >= 4 for s in scores_all]) if scores_all else None,
        "overall_prop_5": safe_prop([s == 5 for s in scores_all]) if scores_all else None,
        # exact counts (useful for micro aggregation)
        "overall_count_ge_4": sum(s >= 4 for s in scores_all),
        "overall_count_5": sum(s == 5 for s in scores_all),
    }

    # Per-category
    per_cat: Dict[int, Dict[str, Any]] = {}
    for cat in TARGET_CATEGORIES:
        cat_rows = [r for r in rows if str(r.get("category", "")).strip() == str(cat)]
        cat_scores = []
        for r in cat_rows:
            sc = parse_score(r.get("score", ""))
            if sc is not None:
                cat_scores.append(sc)

        per_cat[cat] = {
            "attempted_n": len(cat_rows),
            "scored_n": len(cat_scores),
            "parse_fail_n": len(cat_rows) - len(cat_scores),
            "mean_score": safe_mean(cat_scores),
            "prop_ge_4": safe_prop([s >= 4 for s in cat_scores]) if cat_scores else None,
            "prop_5": safe_prop([s == 5 for s in cat_scores]) if cat_scores else None,
            "count_ge_4": sum(s >= 4 for s in cat_scores),
            "count_5": sum(s == 5 for s in cat_scores),
        }

    metrics["per_category"] = per_cat
    return metrics


def mean_ignore_none(vals: List[Any]) -> Optional[float]:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return statistics.mean(nums) if nums else None


def write_outputs(per_file: List[Dict[str, Any]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # CSV rows
    rows: List[Dict[str, Any]] = []
    for entry in per_file:
        sid = entry["sample_id"]
        fname = entry["file"]
        m = entry["metrics"]

        row: Dict[str, Any] = {
            "sample_id": sid,
            "file": fname,
            "overall_attempted_n": m.get("overall_attempted_n"),
            "overall_scored_n": m.get("overall_scored_n"),
            "overall_parse_fail_n": m.get("overall_parse_fail_n"),
            "overall_mean_score": m.get("overall_mean_score"),
            "overall_prop_ge_4": m.get("overall_prop_ge_4"),
            "overall_prop_5": m.get("overall_prop_5"),
        }

        pc = m.get("per_category", {})
        for cat in TARGET_CATEGORIES:
            cm = pc.get(cat, {})
            row[f"cat{cat}_attempted_n"] = cm.get("attempted_n")
            row[f"cat{cat}_scored_n"] = cm.get("scored_n")
            row[f"cat{cat}_parse_fail_n"] = cm.get("parse_fail_n")
            row[f"cat{cat}_mean_score"] = cm.get("mean_score")
            row[f"cat{cat}_prop_ge_4"] = cm.get("prop_ge_4")
            row[f"cat{cat}_prop_5"] = cm.get("prop_5")

        rows.append(row)

    # MACRO_AVG (means/proportions averaged across files; counts summed)
    macro = {"sample_id": "MACRO_AVG", "file": "(macro over files)"}
    keys = list(rows[0].keys()) if rows else []
    for k in keys:
        if k in ("sample_id", "file"):
            continue
        if k.endswith("_attempted_n") or k.endswith("_scored_n") or k.endswith("_parse_fail_n"):
            macro[k] = sum(v for v in (r.get(k) for r in rows) if isinstance(v, int))
        else:
            macro[k] = mean_ignore_none([r.get(k) for r in rows])

    # MICRO_AVG (pooled across all scored rows) using exact counts
    total_attempted = sum(e["metrics"]["overall_attempted_n"] for e in per_file)
    total_scored = sum(e["metrics"]["overall_scored_n"] for e in per_file)
    total_ge_4 = sum(e["metrics"]["overall_count_ge_4"] for e in per_file)
    total_5 = sum(e["metrics"]["overall_count_5"] for e in per_file)

    # pooled mean: weighted by scored_n
    weighted_sum = 0.0
    for e in per_file:
        m = e["metrics"]
        mean = m.get("overall_mean_score")
        n = m.get("overall_scored_n")
        if isinstance(mean, (int, float)) and isinstance(n, int) and n > 0:
            weighted_sum += mean * n

    micro = {
        "sample_id": "MICRO_AVG",
        "file": "(micro over scored QAs)",
        "overall_attempted_n": total_attempted,
        "overall_scored_n": total_scored,
        "overall_parse_fail_n": total_attempted - total_scored,
        "overall_mean_score": (weighted_sum / total_scored) if total_scored > 0 else None,
        "overall_prop_ge_4": (total_ge_4 / total_scored) if total_scored > 0 else None,
        "overall_prop_5": (total_5 / total_scored) if total_scored > 0 else None,
    }

    out_rows = rows + [macro, micro]

    # Write CSV
    csv_path = os.path.join(out_dir, "aba_eval_summary_files_recalc.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    # Write JSON
    json_path = os.path.join(out_dir, "aba_eval_summary_recalc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"per_file": per_file, "macro_average_row": macro, "micro_average_row": micro},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Write TXT
    txt_path = os.path.join(out_dir, "aba_eval_summary_recalc.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ABA Metrics Recalculation (from *_aba_eval.csv)\n")
        f.write("================================================\n\n")
        f.write(f"Files: {len(per_file)}\n")
        f.write(f"Total attempted QAs: {micro['overall_attempted_n']}\n")
        f.write(f"Total scored QAs:    {micro['overall_scored_n']}\n")
        f.write(f"Total parse fails:   {micro['overall_parse_fail_n']}\n\n")
        f.write("MICRO_AVG (pooled over scored QAs):\n")
        f.write(json.dumps(micro, indent=2))
        f.write("\n\nMACRO_AVG (average over files; counts summed):\n")
        f.write(json.dumps(macro, indent=2))
        f.write("\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")


def infer_sample_id_from_csv(rows: List[Dict[str, str]], fallback: str) -> str:
    # Prefer column sample_id if present and consistent; else fallback to filename stem
    sid = rows[0].get("sample_id", "").strip() if rows else ""
    return sid if sid else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Directory containing *_aba_eval.csv files", default="/work/LAS/weile-lab/hbtong/locomo/aba_results")
    ap.add_argument("--out-dir", default=None, help="Where to write summaries (default: results-dir)")
    ap.add_argument("--pattern", default="*_aba_eval.csv", help="Glob pattern (default: *_aba_eval.csv)")
    args = ap.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or results_dir

    csv_files = sorted(glob.glob(os.path.join(results_dir, args.pattern)))
    if not csv_files:
        raise SystemExit(f"No files matched: {os.path.join(results_dir, args.pattern)}")

    per_file: List[Dict[str, Any]] = []
    for csv_path in csv_files:
        rows = load_csv(csv_path)
        fallback_sid = os.path.splitext(os.path.basename(csv_path))[0].replace("_aba_eval", "")
        sample_id = infer_sample_id_f
