#!/usr/bin/env python3
import argparse, os, re, sys
import pandas as pd

FILE_CONV_RE = re.compile(r"conv-(\d+)\.json$", re.IGNORECASE)
EVAL_CONV_RE = re.compile(r"conv-(\d+)_aba_eval\.csv$", re.IGNORECASE)

CAT_TO_COMP = {91: "A", 92: "B", 93: "Aprime"}  # ignore 94


def die(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(1)


def conv_id_from_positions_fileval(fileval: str) -> int:
    m = FILE_CONV_RE.search(str(fileval).strip())
    if not m:
        die(f"Cannot parse conv id from positions 'file' value: {fileval}")
    return int(m.group(1))


def conv_id_from_eval_filename(path: str) -> int:
    base = os.path.basename(path)
    m = EVAL_CONV_RE.search(base)
    if not m:
        die(f"Cannot parse conv id from eval filename: {base}")
    return int(m.group(1))


def assign_ids_positions(pos: pd.DataFrame) -> pd.DataFrame:
    req = {"file", "component", "category", "dia_id", "norm_pos_0_1"}
    miss = req - set(pos.columns)
    if miss:
        die(f"Positions CSV missing required columns: {sorted(miss)}")

    pos = pos.copy()
    pos["conv_id"] = pos["file"].apply(conv_id_from_positions_fileval)
    pos["category"] = pd.to_numeric(pos["category"], errors="coerce").astype("Int64")
    pos["component"] = pos["component"].astype(str).str.strip()

    # preserve original row order
    pos["_row"] = range(len(pos))

    chunks = []
    for conv_id, g in pos.sort_values("_row").groupby("conv_id", sort=True):
        g = g.sort_values("_row").reset_index(drop=True)
        g["fact_seq"] = (g.index // 3).astype(int)
        g["fact_component_id"] = g.apply(
            lambda r: f"{r['component']}_{conv_id}_{int(r['fact_seq'])}", axis=1
        )
        chunks.append(g)

    out = pd.concat(chunks, ignore_index=True).drop(columns=["_row"])
    return out


def assign_ids_eval(ev: pd.DataFrame, conv_id: int) -> pd.DataFrame:
    req = {"index", "category", "question", "evidence_ids", "score"}
    miss = req - set(ev.columns)
    if miss:
        die(f"Eval CSV conv-{conv_id} missing required columns: {sorted(miss)}")

    ev = ev.copy()
    ev["conv_id"] = conv_id
    ev["index"] = pd.to_numeric(ev["index"], errors="coerce")
    if ev["index"].isna().any():
        die(f"Eval CSV conv-{conv_id} has non-numeric 'index' values")

    ev["category"] = pd.to_numeric(ev["category"], errors="coerce").astype("Int64")

    # keep only 91/92/93 for the meta join
    ev = ev[ev["category"].isin([91, 92, 93])].copy()

    ev["fact_seq"] = (ev["index"] // 4).astype(int)
    ev["component"] = ev["category"].map(CAT_TO_COMP)
    ev["fact_component_id"] = ev.apply(
        lambda r: f"{r['component']}_{conv_id}_{int(r['fact_seq'])}", axis=1
    )
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions_csv", required=True)
    ap.add_argument("--aba_results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_name", default="aba_meta_positions_scores.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load + ID positions
    pos = pd.read_csv(args.positions_csv)
    pos2 = assign_ids_positions(pos)

    # Load + ID evals
    eval_paths = sorted(
        os.path.join(args.aba_results_dir, f)
        for f in os.listdir(args.aba_results_dir)
        if EVAL_CONV_RE.match(f)
    )
    if not eval_paths:
        die(f"No conv-*_aba_eval.csv found in {args.aba_results_dir}")

    ev_frames = []
    for p in eval_paths:
        conv_id = conv_id_from_eval_filename(p)
        ev = pd.read_csv(p)
        ev2 = assign_ids_eval(ev, conv_id)
        ev2["eval_csv"] = os.path.basename(p)
        ev_frames.append(ev2)

    ev_all = pd.concat(ev_frames, ignore_index=True)

    # Merge: IMPORTANT suffixes because both sides may have "score"
    meta = pos2.merge(
        ev_all,
        how="left",
        on="fact_component_id",
        suffixes=("_pos", "_eval"),
    )

    # Use eval score explicitly
    # (positions may have score_pos but we want the judge score from eval)
    if "score_eval" in meta.columns:
        meta["score_joined"] = meta["score_eval"]
    elif "score" in meta.columns:
        meta["score_joined"] = meta["score"]
    else:
        meta["score_joined"] = None

    out_csv = os.path.join(args.out_dir, args.out_name)

    keep = [
        "fact_component_id",
        "file", "conv_id_pos", "fact_seq_pos", "component_pos", "category_pos",
        "dia_id", "abs_turn_index", "n_turns", "norm_pos_0_1",
        "eval_csv", "sample_id", "index", "category_eval", "question", "evidence_ids",
        "gold_answer", "model_answer",
        "score_joined",
        "judge_missing", "judge_extra", "judge_explanation", "judge_mode",
    ]
    for c in keep:
        if c not in meta.columns:
            meta[c] = None

    meta[keep].to_csv(out_csv, index=False)

    # Summary / diagnostics
    summary_path = os.path.join(args.out_dir, "aba_meta_positions_scores_summary.txt")
    joined = meta["score_joined"].notna() & (meta["score_joined"].astype(str) != "")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ABA Meta Join Summary\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"positions rows: {len(pos2)}\n")
        f.write(f"eval rows:      {len(ev_all)}\n")
        f.write(f"meta rows:      {len(meta)}\n")
        f.write(f"joined score:   {int(joined.sum())} / {len(meta)}\n\n")

        # Show a few successful joins if any
        ok = meta[joined][["fact_component_id", "score_joined", "question"]].head(10)
        f.write("First 10 successful joins:\n")
        f.write(ok.to_string(index=False) + "\n\n")

        # And a few misses
        miss = meta[~joined][["fact_component_id", "file", "component_pos", "category_pos"]].head(20)
        f.write("First 20 missing joins:\n")
        f.write(miss.to_string(index=False) + "\n")

    print(f"[OK] wrote:\n  {out_csv}\n  {summary_path}")


if __name__ == "__main__":
    main()
