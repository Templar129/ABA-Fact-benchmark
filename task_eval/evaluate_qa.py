# task_eval/evaluate_qa.py
import sys, os, re, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import argparse

from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_aggr_acc

from global_methods import set_openai_key, set_anthropic_key, set_gemini_key
from task_eval.gpt_utils import (
    get_openai_client, get_encoder, count_tokens, truncate_by_tokens_right,
    build_full_context_and_index, chat_once
)
from openai import OpenAI
import google.generativeai as genai

def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out-file', required=True, type=str)
    p.add_argument('--model', required=True, type=str)
    p.add_argument('--data-file', type=str, required=True)
    p.add_argument('--use-rag', action="store_true")
    p.add_argument('--use-4bit', action="store_true")
    p.add_argument('--batch-size', default=1, type=int)
    p.add_argument('--rag-mode', type=str, default="")
    p.add_argument('--emb-dir', type=str, default="")
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--retriever', type=str, default="contriever")
    p.add_argument('--overwrite', action="store_true")
    p.add_argument('--verbose', action="store_true")
    p.add_argument('--log-prompts', action="store_true")
    p.add_argument('--max-context-tokens', type=int, default=128000)
    p.add_argument('--out-root', type=str, default="outputs")
    p.add_argument('--only-category', type=str, default="")
    p.add_argument('--only-these-categories', type=str, default="",
                   help='Comma-separated list, e.g. "91,92,93". Overrides --only-category.')
    p.add_argument('--skip-default-metrics', action="store_true")
    p.add_argument('--use-llm-evaluator', action='store_true')
    p.add_argument('--llm-evaluator-model', type=str, default='gpt-4o-mini')
    p.add_argument('--llm-evaluator-temperature', type=float, default=0.0)
    p.add_argument('--llm-evaluator-max-tokens', type=int, default=32)
    p.add_argument('--llm-evaluator-max-completion-tokens', type=int, default=64,
                   help='For GPT-5-style evaluators that use max_completion_tokens and no temperature.')
    return p.parse_args()

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def build_run_dir(args) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_tag = args.model.replace("/", "_")
    if args.only_these_categories:
        cats = "-".join(sorted({c.strip() for c in args.only_these_categories.split(',') if c.strip()}))
        suffix = f"_cat{cats}"
    elif args.only_category:
        suffix = f"_cat{args.only_category}"
    else:
        suffix = ""
    return ensure_dir(Path(args.out_root) / f"run-{stamp}_{model_tag}{suffix}")

def save_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno())

def append_text(path: Path, text: str):
    with open(path, "a") as f:
        f.write(text + "\n"); f.flush(); os.fsync(f.fileno())

def build_evidence_text(dia_index: dict, evidences: list) -> str:
    if not evidences: return ""
    lines=[dia_index[e] for e in evidences if e in dia_index]
    return "\n".join(lines)

# --- Fairer, semantic-aware evaluator ---
def score_with_llm_evaluator(client: OpenAI, model: str,
                             temp: float, max_toks: int, max_completion_toks: int,
                             q: str, gold: str, ev_text: str, pred: str) -> float:
    system = (
        "You are an impartial grader. Compare the model's answer with the gold answer for factual consistency "
        "based ONLY on the evidence text. Paraphrases that preserve meaning should score close to 1.0. "
        "Partial but essentially correct answers may score 0.6–0.8. "
        "If key facts are wrong or missing, score 0.3 or below. "
        "Return ONLY a number between 0 and 1."
    )
    user = (
        f"Evidence (verbatim):\n{ev_text}\n\n"
        f"Question:\n{q}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Model answer:\n{pred}\n\n"
        "Score (0..1), number only:"
    )
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if model.lower().startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_completion_toks
        else:
            kwargs["temperature"] = temp
            kwargs["max_tokens"] = max_toks

        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"(\d*\.?\d+)", text)
        val = float(m.group(1)) if m else 0.0
        return max(0.0, min(1.0, val))
    except Exception as e:
        print(f"[{ts()}] [evaluator] error: {e} -> score=0.0")
        return 0.0

def main():
    args = parse_args()

    run_dir = build_run_dir(args)
    base = Path(args.out_file).name or "qa_results.json"
    out_json = run_dir / base
    stats_json = run_dir / "qa_stats.json"
    text_log = run_dir / "qa_record.txt"
    eval_json = run_dir / "qa_llm_eval_stats.json"
    meta_json = run_dir / "run_meta.json"

    print(f"[{ts()}] ******************  Evaluating Model {args.model} ***************")
    print(f"[{ts()}] [locomo] data_file={args.data_file}")
    print(f"[{ts()}] [locomo] run_dir={str(run_dir.resolve())}")
    print(f"[{ts()}] [locomo] out_file={str(out_json.resolve())}")

    save_json(meta_json, {
        "model": args.model,
        "data_file": args.data_file,
        "start_time": ts(),
        "max_context_tokens": args.max_context_tokens,
        "use_llm_evaluator": args.use_llm_evaluator,
        "llm_evaluator_model": args.llm_evaluator_model if args.use_llm_evaluator else None,
        "only_category": args.only_category,
        "only_these_categories": args.only_these_categories,
    })
    if not out_json.exists(): save_json(out_json, [])
    if not text_log.exists(): append_text(text_log, f"[{ts()}] (log started)")

    if 'gpt' in args.model:
        set_openai_key()
    elif 'claude' in args.model:
        set_anthropic_key()
    elif 'gemini' in args.model:
        set_gemini_key()
        gmodel = "models/gemini-1.0-pro-latest" if args.model=="gemini-pro-1.0" else args.model
        gemini_model = genai.GenerativeModel(gmodel)
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

    gen_client = get_openai_client() if 'gpt' in args.model else None
    eval_client = OpenAI() if args.use_llm_evaluator else None

    samples = json.load(open(args.data_file))
    print(f"[{ts()}] [locomo] Conversations: {len(samples)}")

    prediction_key = (f"{args.model}_prediction" if not args.use_rag
                      else f"{args.model}_{args.rag_mode}_top_{args.top_k}_prediction")
    model_key = (f"{args.model}" if not args.use_rag
                 else f"{args.model}_{args.rag_mode}_top_{args.top_k}")

    try:
        existing = {d['sample_id']: d for d in json.load(open(out_json))}
    except Exception:
        existing = {}

    enc = get_encoder(args.model)

    with tqdm(total=len(samples), desc="Conversations", unit="conv") as conv_pbar:
        try:
            for cidx, sample in enumerate(samples):
                sample_id = sample.get("sample_id", f"sample_{cidx}")
                print(f"\n[{ts()}] [locomo] ▶ Conversation {cidx+1}/{len(samples)} (sample_id={sample_id})")

                full_context, dia_index = build_full_context_and_index(sample)
                total_tokens = count_tokens(full_context, enc)
                fits = total_tokens <= args.max_context_tokens
                print(f"[{ts()}] [locomo] Conversation tokens={total_tokens} | model window={args.max_context_tokens} | fits={fits}")

                if fits:
                    trimmed_context = full_context; trunc_note = "(no truncation)"
                else:
                    trimmed_context, truncated, before_tok, after_tok = truncate_by_tokens_right(
                        full_context, args.max_context_tokens, enc
                    )
                    trunc_note = f"(truncated {before_tok}→{after_tok})"
                if args.verbose:
                    print(f"[{ts()}] [locomo] Context tokens: {count_tokens(trimmed_context, enc)} {trunc_note}")

                if sample_id in existing:
                    out_data = existing[sample_id]
                else:
                    out_data = {"sample_id": sample_id, "qa": []}

                qa_source = sample.get("qa", [])

                cats = set()
                if args.only_these_categories:
                    cats = {c.strip() for c in args.only_these_categories.split(",") if c.strip()}
                if cats:
                    qa_source = [qa for qa in qa_source if str(qa.get("category")) in cats]
                elif args.only_category:
                    qa_source = [qa for qa in qa_source if str(qa.get("category")) == str(args.only_category)]

                if args.only_these_categories or args.only_category:
                    which = ",".join(sorted(cats)) if cats else args.only_category
                    print(f"[{ts()}] [locomo] Filtered to {len(qa_source)} QAs for categories={which}")
                    if len(qa_source) == 0:
                        raise RuntimeError("No QAs found for the selected categories.")

                done = len(out_data["qa"])

                with tqdm(total=len(qa_source), desc=f"QAs[{sample_id}]", unit="qa", leave=False) as qa_pbar:
                    for qidx, qa in enumerate(qa_source, start=1):
                        if qidx <= done:
                            qa_pbar.update(1); continue

                        question = (qa.get("question") or "").strip()
                        gold = qa.get("answer", "") or ""
                        system_prompt = (
                            "You are a precise assistant. Answer strictly from the provided conversation. "
                            "If unknown, say 'I don't know'. Keep answers concise."
                        )
                        user_prompt = f"{trimmed_context}\n\nQuestion: {question}\nAnswer:"

                        if args.verbose:
                            short_q = (question[:140] + "…") if len(question) > 140 else question
                            print(f"[{ts()}] [locomo] QA {qidx}/{len(qa_source)} — prompting: {short_q}")

                        try:
                            if 'gpt' in args.model:
                                pred = chat_once(
                                    gen_client,
                                    model=args.model,
                                    system=system_prompt,
                                    user=user_prompt,
                                    temperature=0.0,
                                    max_tokens=512
                                )
                            else:
                                raise NotImplementedError("This runner currently supports GPT* generation.")
                        except KeyboardInterrupt:
                            print(f"\n[{ts()}] [locomo] ⚠️  Ctrl-C during generation. Saving partial and exiting.")
                            raise
                        except Exception as e:
                            pred = f"[ERROR] {e}"

                        ev_keys = qa.get("evidence", []) or []
                        ev_text = build_evidence_text(dia_index, ev_keys)

                        row = {
                            "question": question,
                            f"{prediction_key}": pred,
                            "evidence": ev_keys,
                            "category": qa.get("category", None),
                            "subcategory": qa.get("subcategory", None),
                            "topic": qa.get("topic", None),
                            "answer": gold
                        }

                        if args.use_llm_evaluator:
                            score = score_with_llm_evaluator(
                                client=eval_client,
                                model=args.llm_evaluator_model,
                                temp=args.llm_evaluator_temperature,
                                max_toks=args.llm_evaluator_max_tokens,
                                max_completion_toks=args.llm_evaluator_max_completion_tokens,
                                q=question,
                                gold=gold,
                                ev_text=ev_text,
                                pred=pred
                            )
                            row[f"{model_key}_llm_score"] = round(float(score), 3)

                        out_data["qa"].append(row)

                        # persist after each QA
                        existing[sample_id] = out_data
                        save_json(out_json, list(existing.values()))

                        append_text(
                            text_log,
                            f"[{ts()}] sample_id={sample_id} qa#{qidx}\n"
                            f"Q: {question}\nEvidence: {', '.join(ev_keys)}\n"
                            f"GOLD: {gold}\nPRED: {pred}\n"
                            f"Score: {row.get(f'{model_key}_llm_score', '-')}\n---"
                        )

                        qa_pbar.update(1)

                save_json(out_json, list(existing.values()))
                print(f"[{ts()}] [locomo] ✅ Saved conversation {cidx+1}/{len(samples)} → {out_json}")
                conv_pbar.update(1)

        except KeyboardInterrupt:
            print(f"\n[{ts()}] [locomo] ⚠️  Ctrl-C. Partial results saved.")
        except Exception as e:
            print(f"\n[{ts()}] [locomo] ❌ Error: {e}. Partial results saved.")

    save_json(out_json, list(existing.values()))

    if not args.only_category and not args.only_these_categories and not args.skip_default_metrics:
        analyze_aggr_acc(args.data_file, str(out_json), str(stats_json), model_key, model_key + '_f1', rag=args.use_rag)
        print(f"[{ts()}] [locomo] Wrote aggregate stats → {stats_json}")

    preds = json.load(open(out_json))
    sums, counts = {}, {}
    total_sum = 0.0; total_n = 0
    s_key = f"{model_key}_llm_score"
    for sample in preds:
        for row in sample.get("qa", []):
            if s_key not in row: continue
            cat = str(row.get("category","uncat"))
            val = float(row[s_key])
            sums[cat] = sums.get(cat,0.0) + val
            counts[cat] = counts.get(cat,0) + 1
            total_sum += val; total_n += 1

    out_eval = {
        "model": model_key,
        "overall_avg": round(total_sum/total_n, 4) if total_n else 0.0,
        "categories": {cat: {"count": counts[cat],
                             "avg_score": round(sums[cat]/counts[cat], 4)} for cat in sorted(counts)}
    }
    save_json(eval_json, out_eval)
    print(f"[{ts()}] [locomo] Wrote evaluator-based stats → {eval_json}")
    print(f"[{ts()}] [locomo] All artifacts are under: {run_dir.resolve()}")

if __name__ == "__main__":
    main()