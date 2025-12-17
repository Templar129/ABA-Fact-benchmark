#!/usr/bin/env python
"""
Run ABA fact evaluation on long conversations using GPT-5 models.

Key behaviors:
- Answer model sees FULL conversation (with dia_id lines), answers 1 QA at a time (1–2 sentences).
- Judge model sees ONLY evidence lines (with dia_id) + question + gold + model answer.
- Judge output is enforced via Structured Outputs (JSON Schema) to avoid missing SCORE.

Structured Outputs with Responses API:
text: { format: { type: "json_schema", strict: true, schema: ... } }
Docs: https://platform.openai.com/docs/guides/structured-outputs
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Any, Tuple, Optional

from openai import OpenAI  # type: ignore


TARGET_CATEGORIES = {91, 92, 93, 94}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_conversation_view(conv_obj: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """
    Flatten sessions into:
      - conversation_text: long string with "dia_id speaker: text" per line
      - dia_map: dia_id -> "dia_id speaker: text"
    """
    session_keys = [k for k in conv_obj.keys() if k.startswith("session_") and not k.endswith("_date_time")]

    def session_index(k: str) -> int:
        try:
            return int(k.split("_")[1])
        except Exception:
            return 999_999

    session_keys.sort(key=session_index)

    dia_map: Dict[str, str] = {}
    lines: List[str] = []

    for sk in session_keys:
        turns = conv_obj[sk]
        for turn in turns:
            dia_id = str(turn.get("dia_id", "")).strip()
            speaker = str(turn.get("speaker", "")).strip()
            text = str(turn.get("text", "")).strip()
            if not dia_id:
                continue
            line = f"{dia_id} {speaker}: {text}" if speaker else f"{dia_id}: {text}"
            dia_map[dia_id] = line
            lines.append(line)

    return "\n".join(lines), dia_map


def filter_aba_qas(qa_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [qa for qa in qa_list if qa.get("category") in TARGET_CATEGORIES]


def get_gold_answer(qa: Dict[str, Any]) -> str:
    # Some datasets use "answer", others "adversarial_answer"
    return qa.get("adversarial_answer", qa.get("answer", ""))


def call_answer_model(
    client: OpenAI,
    model: str,
    conversation_text: str,
    question: str,
    max_output_tokens: int = 128,
) -> str:
    system_msg = (
        "You are a careful reader of long conversations. "
        "Answer questions using only the information in the conversation. "
        "If the answer is not determinable from the conversation, say "
        "'I cannot tell from the conversation.'"
    )

    user_content = f"""You are answering a question about the following conversation.

CONVERSATION:
{conversation_text}

QUESTION:
{question}

Please answer the question in 1–2 sentences. Do not explain your reasoning.
If the answer cannot be determined from the conversation, respond with:
"I cannot tell from the conversation."
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=max_output_tokens,
    )
    return (resp.output_text or "").strip()


# ---------- Judge (Structured Outputs) ----------

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "description": "Integer score from 0 to 5 inclusive.",
        },
        "missing": {
            "type": "string",
            "description": "Key information from GOLD that is missing in the model answer, or 'NONE'.",
        },
        "extra": {
            "type": "string",
            "description": "Extra claims in model answer not present in GOLD, or 'NONE'.",
        },
        "explanation": {
            "type": "string",
            "description": "1–3 sentences justifying the score.",
        },
    },
    "required": ["score", "missing", "extra", "explanation"],
}


def _fallback_parse_score_text(text: str) -> Dict[str, Any]:
    """
    Emergency fallback (should be rare) if the model fails structured output.
    """
    def extract_line(prefix: str) -> Optional[str]:
        m = re.search(rf"^{prefix}\s*:\s*(.*)$", text, flags=re.MULTILINE)
        return m.group(1).strip() if m else None

    score_str = extract_line("SCORE")
    score = int(score_str) if score_str and re.fullmatch(r"[0-5]", score_str) else None
    return {
        "score": score,
        "missing": extract_line("MISSING") or "",
        "extra": extract_line("EXTRA") or "",
        "explanation": extract_line("EXPLANATION") or "",
        "raw": text,
        "mode": "fallback_text",
    }


def call_judge_model(
    client: OpenAI,
    model: str,
    evidence_text: str,
    question: str,
    model_answer: str,
    gold_answer: str,
    max_output_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Strict judge with schema-enforced output so SCORE is always present and parseable.

    Scoring emphasis:
    - Judge must consider EVIDENCE to interpret what the question refers to.
    - Then compare MODEL ANSWER vs GOLD ANSWER semantically.
    - 5 requires perfect semantic match with no extra claims and no missing key facts.
    """

    system_msg = "You are an exacting evaluator for QA over conversations. Grade strictly and consistently."

    # Keep the instruction concise; schema enforces format.
    user_content = f"""Grade the MODEL ANSWER against the GOLD ANSWER, using EVIDENCE to interpret context.

Scoring (0–5):
5 = perfect semantic match to GOLD; no extra claims; no missing key facts.
4 = basically correct; only minor vagueness or tiny omission.
3 = mostly correct; missing a key detail OR includes a minor extra claim.
2 = partially correct; significant omissions and/or significant extra/wrong claims.
1 = mostly wrong; only small overlap with GOLD.
0 = completely wrong OR contradicts GOLD/evidence.

Rules:
- Penalize EXTRA information not in GOLD.
- Penalize MISSING key information from GOLD.
- Use EVIDENCE only for context; do not “fix” GOLD.

EVIDENCE:
{evidence_text}

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

MODEL ANSWER:
{model_answer}
"""

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            # Structured Outputs (Responses API):
            text={
                "format": {
                    "type": "json_schema",
                    "name": "aba_judge_result",
                    "strict": True,
                    "schema": JUDGE_SCHEMA,
                }
            },
            max_output_tokens=max_output_tokens,
        )

        raw = (resp.output_text or "").strip()

        # With structured outputs, output_text should be JSON. Parse it.
        parsed = json.loads(raw)

        # Defensive normalization
        score = parsed.get("score", None)
        if not isinstance(score, int) or not (0 <= score <= 5):
            score = None

        missing = str(parsed.get("missing", "")).strip()
        extra = str(parsed.get("extra", "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()

        return {
            "score": score,
            "missing": missing,
            "extra": extra,
            "explanation": explanation,
            "raw": raw,
            "mode": "json_schema",
        }

    except Exception as e:
        # Fallback: ask for the old 4-line format (rare)
        fallback_prompt = user_content + """

Respond EXACTLY in this format (4 lines):
SCORE: <integer 0-5>
MISSING: <NONE or short phrase>
EXTRA: <NONE or short phrase>
EXPLANATION: <1-3 sentences>
"""
        resp2 = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": fallback_prompt},
            ],
            max_output_tokens=max_output_tokens,
        )
        text2 = (resp2.output_text or "").strip()
        out = _fallback_parse_score_text(text2)
        out["error"] = f"Structured output failed: {repr(e)}"
        return out


# ---------- Metrics / Saving ----------

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores_all = [r["score"] for r in results if isinstance(r.get("score"), int)]
    metrics: Dict[str, Any] = {}

    metrics["overall_attempted_n"] = len(results)
    metrics["overall_scored_n"] = len(scores_all)
    metrics["overall_parse_fail_n"] = len(results) - len(scores_all)

    if scores_all:
        metrics["overall_mean_score"] = statistics.mean(scores_all)
        metrics["overall_prop_ge_4"] = sum(s >= 4 for s in scores_all) / len(scores_all)
        metrics["overall_prop_5"] = sum(s == 5 for s in scores_all) / len(scores_all)
    else:
        metrics["overall_mean_score"] = None
        metrics["overall_prop_ge_4"] = None
        metrics["overall_prop_5"] = None

    per_cat: Dict[int, Dict[str, Any]] = {}
    for cat in sorted(TARGET_CATEGORIES):
        cat_attempted = [r for r in results if r.get("category") == cat]
        cat_scores = [r["score"] for r in cat_attempted if isinstance(r.get("score"), int)]
        per_cat[cat] = {
            "attempted_n": len(cat_attempted),
            "scored_n": len(cat_scores),
            "parse_fail_n": len(cat_attempted) - len(cat_scores),
            "mean_score": statistics.mean(cat_scores) if cat_scores else None,
            "prop_ge_4": (sum(s >= 4 for s in cat_scores) / len(cat_scores)) if cat_scores else None,
            "prop_5": (sum(s == 5 for s in cat_scores) / len(cat_scores)) if cat_scores else None,
        }

    metrics["per_category"] = per_cat
    return metrics


def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any], sample_id: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    out_json = {"sample_id": sample_id, "metrics": metrics, "results": results}
    json_path = os.path.join(output_dir, f"{sample_id}_aba_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, f"{sample_id}_aba_eval.csv")
    fieldnames = [
        "sample_id",
        "index",
        "category",
        "question",
        "gold_answer",
        "evidence_ids",
        "model_answer",
        "score",
        "judge_missing",
        "judge_extra",
        "judge_explanation",
        "judge_mode",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "index": r.get("index"),
                    "category": r.get("category"),
                    "question": r.get("question"),
                    "gold_answer": r.get("gold_answer"),
                    "evidence_ids": ";".join(r.get("evidence_ids", [])),
                    "model_answer": r.get("model_answer"),
                    "score": r.get("score"),
                    "judge_missing": r.get("judge_missing"),
                    "judge_extra": r.get("judge_extra"),
                    "judge_explanation": r.get("judge_explanation"),
                    "judge_mode": r.get("judge_mode"),
                }
            )

    print(f"Saved JSON results to: {json_path}")
    print(f"Saved CSV results to:  {csv_path}")


def run_on_file(
    client: OpenAI,
    path: str,
    answer_model: str,
    judge_model: str,
    output_dir: str,
    answer_max_tokens: int,
    judge_max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    data = load_json(path)
    sample_id = data.get("sample_id", os.path.splitext(os.path.basename(path))[0])

    conv_obj = data["conversation"]
    qa_list = data["qa"]

    conversation_text, dia_map = build_conversation_view(conv_obj)
    aba_qas = filter_aba_qas(qa_list)

    print(f"File: {path}")
    print(f"Sample ID: {sample_id}")
    print(f"Total QA entries: {len(qa_list)}")
    print(f"ABA QA entries (91–94): {len(aba_qas)}")
    print("Running evaluation...\n")

    results: List[Dict[str, Any]] = []

    for idx, qa in enumerate(aba_qas):
        question = qa["question"]
        gold_answer = get_gold_answer(qa)
        category = qa.get("category")
        evidence_ids: List[str] = qa.get("evidence", [])

        print(f"### QA {idx+1}/{len(aba_qas)} (category={category})")
        print(f"Question: {question}")

        model_answer = call_answer_model(
            client=client,
            model=answer_model,
            conversation_text=conversation_text,
            question=question,
            max_output_tokens=answer_max_tokens,
        )
        print(f"Model answer: {model_answer}")

        evidence_lines = []
        for eid in evidence_ids:
            line = dia_map.get(str(eid))
            if line is not None:
                evidence_lines.append(line)
        evidence_text = "\n".join(evidence_lines)

        judge_result = call_judge_model(
            client=client,
            model=judge_model,
            evidence_text=evidence_text,
            question=question,
            model_answer=model_answer,
            gold_answer=gold_answer,
            max_output_tokens=judge_max_tokens,
        )

        print(f"Judge score: {judge_result['score']}")
        print(f"Judge missing: {judge_result.get('missing')}")
        print(f"Judge extra: {judge_result.get('extra')}")
        print(f"Judge explanation: {judge_result.get('explanation')}")
        print(f"Judge mode: {judge_result.get('mode')}\n")

        results.append(
            {
                "index": idx,
                "category": category,
                "question": question,
                "gold_answer": gold_answer,
                "evidence_ids": evidence_ids,
                "evidence_text": evidence_text,
                "model_answer": model_answer,
                "score": judge_result.get("score"),
                "judge_missing": judge_result.get("missing"),
                "judge_extra": judge_result.get("extra"),
                "judge_explanation": judge_result.get("explanation"),
                "judge_raw": judge_result.get("raw"),
                "judge_mode": judge_result.get("mode"),
                "judge_error": judge_result.get("error"),
            }
        )

    metrics = compute_metrics(results)
    print("=== Summary metrics ===")
    print(json.dumps(metrics, indent=2))
    print()

    save_results(results, metrics, sample_id, output_dir)
    return sample_id, metrics


def write_summary_over_files(per_file: List[Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

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
        for cat in sorted(TARGET_CATEGORIES):
            cm = pc.get(cat, {})
            row[f"cat{cat}_attempted_n"] = cm.get("attempted_n")
            row[f"cat{cat}_scored_n"] = cm.get("scored_n")
            row[f"cat{cat}_parse_fail_n"] = cm.get("parse_fail_n")
            row[f"cat{cat}_mean_score"] = cm.get("mean_score")
            row[f"cat{cat}_prop_ge_4"] = cm.get("prop_ge_4")
            row[f"cat{cat}_prop_5"] = cm.get("prop_5")
        rows.append(row)

    def mean_ignore_none(values: List[Any]) -> Any:
        nums = [v for v in values if isinstance(v, (int, float))]
        return statistics.mean(nums) if nums else None

    macro_row: Dict[str, Any] = {"sample_id": "MACRO_AVG", "file": "(macro over files)"}
    for key in rows[0].keys():
        if key in ("sample_id", "file"):
            continue
        if key.endswith("_attempted_n") or key.endswith("_scored_n") or key.endswith("_parse_fail_n"):
            macro_row[key] = sum(v for v in (r.get(key) for r in rows) if isinstance(v, int))
        else:
            macro_row[key] = mean_ignore_none([r.get(key) for r in rows])

    rows_with_macro = rows + [macro_row]

    csv_path = os.path.join(output_dir, "aba_eval_summary_files.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_with_macro[0].keys()))
        writer.writeheader()
        for r in rows_with_macro:
            writer.writerow(r)
    print(f"Saved summary CSV to: {csv_path}")

    json_path = os.path.join(output_dir, "aba_eval_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"per_file": per_file, "macro_average_row": macro_row}, f, indent=2, ensure_ascii=False)
    print(f"Saved summary JSON to: {json_path}")

    txt_path = os.path.join(output_dir, "aba_eval_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ABA Evaluation Summary Across Files\n")
        f.write("===================================\n\n")
        for entry in per_file:
            sid = entry["sample_id"]
            fname = entry["file"]
            m = entry["metrics"]
            f.write(f"File: {fname}\n")
            f.write(f"Sample ID: {sid}\n")
            f.write(f"  attempted_n: {m.get('overall_attempted_n')}\n")
            f.write(f"  scored_n:    {m.get('overall_scored_n')}\n")
            f.write(f"  parse_fail:  {m.get('overall_parse_fail_n')}\n")
            f.write(f"  mean_score:  {m.get('overall_mean_score')}\n")
            f.write(f"  prop_ge_4:   {m.get('overall_prop_ge_4')}\n")
            f.write(f"  prop_5:      {m.get('overall_prop_5')}\n\n")
        f.write("Macro row:\n")
        f.write(json.dumps(macro_row, indent=2))
        f.write("\n")
    print(f"Saved summary TXT to: {txt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run ABA fact evaluation on conversation JSON files.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=str, help="A JSON file path OR a directory containing JSON files.")
    g.add_argument("--inputs", nargs="+", help="Explicit list of JSON files to run, in order.")

    p.add_argument("--output-dir", type=str, default="./aba_eval_outputs", help="Output directory for results.")
    p.add_argument("--answer-model", type=str, default="gpt-5-mini")
    p.add_argument("--judge-model", type=str, default="gpt-5-mini")
    p.add_argument("--answer-max-tokens", type=int, default=2048)
    p.add_argument("--judge-max-tokens", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()
    client = OpenAI()

    if args.inputs:
        json_files = args.inputs
    else:
        input_path = args.input
        if os.path.isdir(input_path):
            json_files = sorted(glob.glob(os.path.join(input_path, "*.json")))
        else:
            json_files = [input_path]

    if not json_files:
        print("No JSON files to process.")
        return

    print(f"Will process {len(json_files)} file(s).\n")

    all_metrics: List[Dict[str, Any]] = []
    for path in json_files:
        sample_id, metrics = run_on_file(
            client=client,
            path=path,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            output_dir=args.output_dir,
            answer_max_tokens=args.answer_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
        )
        all_metrics.append({"sample_id": sample_id, "file": os.path.basename(path), "metrics": metrics})

    write_summary_over_files(all_metrics, args.output_dir)


if __name__ == "__main__":
    main()
