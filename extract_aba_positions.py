#!/usr/bin/env python3
"""
extract_chatgpt_positions_scores.py

Goal:
- Extract relative (0..1) positions of all ABA' fact components across 10 JSON files.
- For each component (A from cat91, B from cat92, A' from cat93), also extract the QA score.

Method:
- Build dia_id order from conversation["session_*"][i]["dia_id"].
- For each QA in cat 91/92/93:
  - anchor = earliest dia_id in evidence[] that exists in dia_index
  - score = first numeric score found among common keys (see SCORE_KEYS)
- Sort QAs within each category by anchor index
- Align by rank (i-th in cat91/92/93 is one fact triple)
- Emit 3 rows per fact rank: (A,B,A').

Outputs:
- <out_prefix>_fact_components.csv
- <out_prefix>_fact_components_summary.txt
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


DIA_RE = re.compile(r"^D(\d+):(\d+)$", re.IGNORECASE)
SESSION_RE = re.compile(r"^session_(\d+)$", re.IGNORECASE)

# Add/remove keys here to match whatever your pipeline wrote.
SCORE_KEYS = [
    "score",
    "points",
    "grade",
    "rating",
    "judge_score",
    "grader_score",
    "gpt_score",
    "final_score",
    "overall_score",
    "eval_score",
    "correctness_score",
    "model_score",
    "qa_score",
]


def parse_dia_id(s: str) -> bool:
    return isinstance(s, str) and DIA_RE.match(s.strip()) is not None


def sorted_session_keys(conversation: Dict[str, Any]) -> List[str]:
    keys = []
    for k in conversation.keys():
        m = SESSION_RE.match(k)
        if m:
            keys.append((int(m.group(1)), k))
    keys.sort()
    return [k for _, k in keys]


def extract_dia_sequence(conversation: Dict[str, Any]) -> List[str]:
    seq: List[str] = []
    for sk in sorted_session_keys(conversation):
        sess = conversation.get(sk)
        if not isinstance(sess, list):
            continue
        for turn in sess:
            if isinstance(turn, dict):
                d = turn.get("dia_id")
                if isinstance(d, str) and parse_dia_id(d):
                    seq.append(d.strip())
    return seq


def build_index(seq: List[str]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for i, d in enumerate(seq):
        if d not in m:
            m[d] = i
    return m


def norm_pos(i: int, n: int) -> float:
    if n <= 1:
        return 0.0
    return i / (n - 1)


def get_category(q: Dict[str, Any]) -> Optional[int]:
    # most likely: q["category"]
    v = q.get("category")
    if isinstance(v, int):
        return v
    try:
        if v is not None:
            return int(v)
    except Exception:
        pass
    # fallback: nested meta
    meta = q.get("meta")
    if isinstance(meta, dict):
        v2 = meta.get("category")
        if isinstance(v2, int):
            return v2
        try:
            if v2 is not None:
                return int(v2)
        except Exception:
            pass
    return None


def extract_score(q: Dict[str, Any]) -> Optional[float]:
    """
    Search for a numeric score in common locations.
    If your score is nested, we also check q.get("meta") and q.get("grading") etc.
    """
    def try_val(x: Any) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            x = x.strip()
            # allow "3/5" or "score: 4" patterns
            m = re.search(r"(-?\d+(?:\.\d+)?)", x)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
        return None

    # direct keys
    for k in SCORE_KEYS:
        if k in q:
            sc = try_val(q.get(k))
            if sc is not None:
                return sc

    # common nested containers
    for container_key in ("meta", "grading", "grader", "judge", "evaluation", "eval"):
        c = q.get(container_key)
        if isinstance(c, dict):
            for k in SCORE_KEYS:
                if k in c:
                    sc = try_val(c.get(k))
                    if sc is not None:
                        return sc

    # last resort: scan shallow string fields for a number labeled score
    for k, v in q.items():
        if isinstance(v, str) and ("score" in v.lower() or "points" in v.lower()):
            sc = try_val(v)
            if sc is not None:
                return sc

    return None


def evidence_anchor(q: Dict[str, Any], dia_index: Dict[str, int]) -> Optional[Tuple[int, str]]:
    ev = q.get("evidence")
    if not isinstance(ev, list) or not ev:
        return None
    hits = []
    for x in ev:
        if isinstance(x, str):
            d = x.strip()
            if d in dia_index:
                hits.append((dia_index[d], d))
    if not hits:
        return None
    return min(hits)  # earliest in conversation


def summarize(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": float(mean(xs)),
        "median": float(median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", default="conv-*.json")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched: {os.path.join(args.input_dir, args.pattern)}")

    out_csv = f"{args.out_prefix}_fact_components.csv"
    out_txt = f"{args.out_prefix}_fact_components_summary.txt"

    rows: List[Dict[str, Any]] = []
    global_diag = Counter()

    pooled_pos = {"A": [], "B": [], "Aprime": []}
    pooled_scores = {"A": [], "B": [], "Aprime": []}

    per_file_blocks: List[str] = []

    for path in paths:
        fname = os.path.basename(path)
        obj = json.load(open(path, "r", encoding="utf-8"))

        conversation = obj.get("conversation")
        qa = obj.get("qa")

        issues = Counter()

        if not isinstance(conversation, dict):
            issues["conversation_not_dict"] += 1
            global_diag.update(issues)
            continue
        if not isinstance(qa, list):
            issues["qa_not_list"] += 1
            global_diag.update(issues)
            continue

        dia_seq = extract_dia_sequence(conversation)
        n_turns = len(dia_seq)
        if n_turns == 0:
            issues["no_dia_ids_found"] += 1
            global_diag.update(issues)
            continue

        dia_index = build_index(dia_seq)

        # Collect items per category with anchors
        items = {91: [], 92: [], 93: []}  # list of (anchor_idx, anchor_dia, score, qa_obj)
        for q in qa:
            if not isinstance(q, dict):
                continue
            cat = get_category(q)
            if cat not in (91, 92, 93):
                continue

            anch = evidence_anchor(q, dia_index)
            if anch is None:
                issues[f"cat{cat}_no_valid_anchor"] += 1
                continue
            a_idx, a_dia = anch

            sc = extract_score(q)
            if sc is None:
                issues[f"cat{cat}_missing_score"] += 1

            items[cat].append((a_idx, a_dia, sc, q))

        for cat in (91, 92, 93):
            items[cat].sort(key=lambda x: x[0])

        n91, n92, n93 = len(items[91]), len(items[92]), len(items[93])
        k = min(n91, n92, n93)

        if k == 0:
            issues["no_complete_triplets_by_rank"] += 1

        order_ok = 0
        order_bad = 0

        # Create 3 rows per fact rank
        for i in range(k):
            a_idx, a_dia, a_sc, _ = items[91][i]
            b_idx, b_dia, b_sc, _ = items[92][i]
            p_idx, p_dia, p_sc, _ = items[93][i]

            a_pos = norm_pos(a_idx, n_turns)
            b_pos = norm_pos(b_idx, n_turns)
            p_pos = norm_pos(p_idx, n_turns)

            ok = (a_idx <= b_idx <= p_idx)
            if ok:
                order_ok += 1
            else:
                order_bad += 1

            def add_row(component: str, cat: int, dia: str, idx: int, pos: float, sc: Optional[float]):
                rows.append({
                    "file": fname,
                    "fact_rank": i,
                    "component": component,
                    "category": cat,
                    "dia_id": dia,
                    "abs_turn_index": idx,
                    "n_turns": n_turns,
                    "norm_pos_0_1": f"{pos:.6f}",
                    "score": "" if sc is None else sc,
                    "order_ok_triplet": ok,
                })
                pooled_pos[component].append(pos)
                if sc is not None:
                    pooled_scores[component].append(sc)

            add_row("A", 91, a_dia, a_idx, a_pos, a_sc)
            add_row("B", 92, b_dia, b_idx, b_pos, b_sc)
            add_row("Aprime", 93, p_dia, p_idx, p_pos, p_sc)

        # Per-file summaries
        block = []
        block.append(f"File: {fname}")
        block.append(f"  n_turns_with_dia_id: {n_turns}")
        block.append(f"  qa anchors found: cat91={n91}, cat92={n92}, cat93={n93}")
        block.append(f"  complete_triplets_by_rank: {k} (order_ok={order_ok}, order_bad={order_bad})")

        # positions per file
        Apos = [float(r["norm_pos_0_1"]) for r in rows if r["file"] == fname and r["component"] == "A"]
        Bpos = [float(r["norm_pos_0_1"]) for r in rows if r["file"] == fname and r["component"] == "B"]
        Ppos = [float(r["norm_pos_0_1"]) for r in rows if r["file"] == fname and r["component"] == "Aprime"]
        block.append(f"  A pos: {summarize(Apos)}")
        block.append(f"  B pos: {summarize(Bpos)}")
        block.append(f"  A' pos: {summarize(Ppos)}")

        if issues:
            block.append(f"  issues: {dict(issues)}")

        per_file_blocks.append("\n".join(block))
        global_diag.update(issues)

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "file", "fact_rank", "component", "category",
                "dia_id", "abs_turn_index", "n_turns", "norm_pos_0_1",
                "score", "order_ok_triplet"
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write TXT summary
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("ABA Fact Component Positions + Scores Summary\n")
        f.write("=" * 72 + "\n\n")
        f.write("Pooled component position stats (over all extracted rows):\n")
        f.write(f"  - A:      {summarize(pooled_pos['A'])}\n")
        f.write(f"  - B:      {summarize(pooled_pos['B'])}\n")
        f.write(f"  - A':     {summarize(pooled_pos['Aprime'])}\n\n")

        f.write("Pooled component score stats (only where score present):\n")
        f.write(f"  - A:      {summarize(pooled_scores['A'])}\n")
        f.write(f"  - B:      {summarize(pooled_scores['B'])}\n")
        f.write(f"  - A':     {summarize(pooled_scores['Aprime'])}\n\n")

        f.write("Per-file summaries:\n")
        f.write("-" * 72 + "\n\n")
        f.write("\n\n".join(per_file_blocks) + "\n\n")

        f.write("Global diagnostics:\n")
        f.write(f"  {dict(global_diag)}\n\n")

        f.write("Notes:\n")
        f.write("  - One output row corresponds to one (fact_rank, component).\n")
        f.write("  - Triplets are aligned by sorting each category by anchor dia position and aligning ranks.\n")
        f.write("  - score is extracted from common keys; adjust SCORE_KEYS if your field name differs.\n")

    print(f"Wrote:\n  {out_csv}\n  {out_txt}")


if __name__ == "__main__":
    main()
