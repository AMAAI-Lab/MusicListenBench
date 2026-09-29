"""Score MusicCaps caption predictions (Probe B, easy version): corpus + per-clip
captioning metrics for each (model, variant), and paired base-vs-trained deltas.

Metrics
-------
* BLEU-4 (corpus, NLTK corpus_bleu) + smoothed sentence BLEU-4 (per-clip)
* METEOR (NLTK, per-clip; corpus value = mean, standard for single-reference)
* ROUGE-L (per-clip LCS F1, beta=1.2; corpus value = mean)
* BERTScore-F1 (bert-score, roberta-large, rescale_with_baseline=True) — skipped
  automatically with a warning if the package/backbone is unavailable
* Length / degeneracy diagnostics (mean words, share <5 words, share degenerate
  A/B/yes/no/same/different answers, share hitting max_new_tokens)

Stats (paired: every checkpoint captions the SAME clips)
* per-clip deltas (sBLEU, METEOR, ROUGE-L, BERTScore, words): mean delta with
  10k-draw bootstrap CI + Wilcoxon signed-rank p (Holm-corrected per family over
  metric x variant)
* corpus BLEU delta: paired bootstrap CI (2k draws)

Run after run_all_captioning.sh (or run_captioning.py runs):
    python score_captions.py            # -> results/summary.md + results/summary.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from musiclistenbench import paths

RESULTS = Path(paths.RESULTS_DIR) / "captioning"

FAMILIES = {  # family tag -> variant order; base row is the reference
    "qwen2_audio": ["base", "lora", "fullft"],
    "qwen2_5_omni": ["base", "lora", "fullft"],
}
DEGENERATE = {"a", "b", "yes", "no", "same", "different", "(a)", "(b)", "answer:"}


# ----------------------------- tokenisation / metrics -----------------------------

def toks(s: str):
    from nltk import word_tokenize
    return word_tokenize(s.lower())


def sentence_bleu(ref_toks, hyp_toks) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu as sb
    if not hyp_toks:
        return 0.0
    return sb([ref_toks], hyp_toks, smoothing_function=SmoothingFunction().method1)


def corpus_bleu(refs_list, hyps_list) -> float:
    from nltk.translate.bleu_score import corpus_bleu as cb
    try:
        return cb([[r] for r in refs_list], hyps_list)
    except ZeroDivisionError:
        return 0.0


def meteor(ref_toks, hyp_toks) -> float:
    from nltk.translate.meteor_score import meteor_score as ms
    if not hyp_toks:
        return 0.0
    return ms([ref_toks], hyp_toks)


def lcs_len(a, b) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l(ref_toks, hyp_toks, beta=1.2) -> float:
    if not ref_toks or not hyp_toks:
        return 0.0
    k = lcs_len(ref_toks, hyp_toks)
    p, r = k / len(hyp_toks), k / len(ref_toks)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def is_degenerate(gen: str) -> bool:
    g = gen.strip().lower().strip(".!\"' ")
    if g in DEGENERATE:
        return True
    if len(g.split()) == 1 and g.rstrip(").") in {"a", "b", "yes", "no", "same", "different"}:
        return True
    return False


# ------------------------------- paired statistics --------------------------------

def boot_ci(deltas: np.ndarray, n_boot=10000, seed=0):
    """Mean delta with percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = deltas[idx].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def stable_seed(*parts) -> int:
    import zlib
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2**31)


def wilcoxon_p(deltas: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = deltas[deltas != 0]
    if len(d) < 5:
        return float("nan")
    try:
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def holm(pvals):
    order = np.argsort(np.asarray(pvals, dtype=float))
    m = len(pvals)
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * float(pvals[i]))
        running = max(running, v)
        adj[i] = running
    return adj


# ------------------------------------- main ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds-dir", default=str(RESULTS))
    ap.add_argument("--out-prefix", default=str(RESULTS / "summary"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-boot-bleu", type=int, default=2000)
    ap.add_argument("--device", default=None, help="bert-score device (default: cuda if free)")
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for fam, variants in FAMILIES.items():
        for v in variants:
            p = preds_dir / f"preds_{fam}_{v}.jsonl"
            if p.exists():
                files[(fam, v)] = p
    if not files:
        raise SystemExit(f"no preds_*.jsonl found under {preds_dir}")

    data = {}
    for key, p in files.items():
        recs = {}
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[r["ytid"]] = r
        data[key] = recs
        print(f"loaded {p.name}: {len(recs)} captions")

    # paired evaluation set: clips captioned by EVERY loaded config
    common = set.intersection(*(set(recs) for recs in data.values()))
    common = sorted(common)
    if len(common) < 10:
        raise SystemExit(f"too few common clips across configs: {len(common)}")
    print(f"paired eval set: {len(common)} common clips")

    refs = [data[next(iter(files))][y]["ref"] for y in common]
    ref_toks = [toks(r) for r in refs]

    # ---- BERTScore (all configs at once, one backbone) ----
    bs_f1 = {}
    use_bs = not args.no_bertscore
    if use_bs:
        try:
            import torch
            from bert_score import score as bs_score
            device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            for key, recs in data.items():
                cands = [recs[y]["gen"] for y in common]
                _, _, F = bs_score(cands, refs, model_type="roberta-large", lang="en",
                                   rescale_with_baseline=True, batch_size=64,
                                   device=device)
                bs_f1[key] = F.numpy().astype(float)
                print(f"BERTScore done for {key}")
        except Exception as e:
            print(f"!! BERTScore unavailable ({e}); continuing without it")
            bs_f1 = {}

    # ---- per-clip metrics + per-config summaries ----
    perclip_rows = []
    summaries = {}
    for key, recs in data.items():
        gens = [recs[y]["gen"] for y in common]
        hyp_toks_l = [toks(g) if g.strip() else [] for g in gens]
        s_bleu = np.array([sentence_bleu(rt, ht) for rt, ht in zip(ref_toks, hyp_toks_l)])
        met = np.array([meteor(rt, ht) for rt, ht in zip(ref_toks, hyp_toks_l)])
        rl = np.array([rouge_l(rt, ht) for rt, ht in zip(ref_toks, hyp_toks_l)])
        words = np.array([len(g.split()) for g in gens])
        n_hit = sum(bool(recs[y].get("hit_max_new_tokens", False)) for y in common)
        n_degen = sum(is_degenerate(g) for g in gens)
        n_empty = sum(not g.strip() for g in gens)

        summaries[key] = {
            "n": len(common),
            "bleu4_corpus": corpus_bleu(ref_toks, [h for h in hyp_toks_l]),
            "sbleu_mean": float(s_bleu.mean()),
            "meteor_mean": float(met.mean()),
            "rougeL_mean": float(rl.mean()),
            "bertscore_f1_mean": float(bs_f1[key].mean()) if key in bs_f1 else None,
            "words_mean": float(words.mean()),
            "words_median": float(np.median(words)),
            "pct_lt5_words": float((words < 5).mean() * 100),
            "pct_degenerate": float(n_degen / len(common) * 100),
            "pct_empty": float(n_empty / len(common) * 100),
            "pct_hit_max_new_tokens": float(n_hit / len(common) * 100),
        }
        for j, y in enumerate(common):
            perclip_rows.append({
                "ytid": y, "model": key[0], "variant": key[1],
                "sbleu": float(s_bleu[j]), "meteor": float(met[j]),
                "rougeL": float(rl[j]),
                "bertscore_f1": float(bs_f1[key][j]) if key in bs_f1 else None,
                "words": int(words[j]),
            })

    # cache per-clip arrays for the paired section
    arr = {}
    for key in data:
        rows = [r for r in perclip_rows if (r["model"], r["variant"]) == key]
        arr[key] = {
            "sbleu": np.array([r["sbleu"] for r in rows]),
            "meteor": np.array([r["meteor"] for r in rows]),
            "rougeL": np.array([r["rougeL"] for r in rows]),
            "words": np.array([r["words"] for r in rows], dtype=float),
        }
        if key in bs_f1:
            arr[key]["bertscore_f1"] = bs_f1[key]

    # ---- paired deltas vs base, per family ----
    deltas = {}
    for fam, variants in FAMILIES.items():
        if (fam, "base") not in data:
            continue
        for v in variants:
            if v == "base" or (fam, v) not in data:
                continue
            key_t, key_b = (fam, v), (fam, "base")
            entry = {"per_metric": {}}
            raw_p = []
            for m in (["sbleu", "meteor", "rougeL", "bertscore_f1"]
                      if "bertscore_f1" in arr[key_t] else ["sbleu", "meteor", "rougeL"]):
                d = arr[key_t][m] - arr[key_b][m]
                lo, hi = boot_ci(d, n_boot=args.n_boot, seed=stable_seed(fam, v, m))
                p = wilcoxon_p(d)
                entry["per_metric"][m] = {
                    "mean_delta": float(d.mean()),
                    "median_delta": float(np.median(d)),
                    "ci95": [lo, hi],
                    "wilcoxon_p_raw": p,
                }
                raw_p.append(p)
            # words delta (diagnostic, no test)
            dw = arr[key_t]["words"] - arr[key_b]["words"]
            entry["words_mean_delta"] = float(dw.mean())
            entry["words_ci95"] = list(boot_ci(dw, n_boot=args.n_boot, seed=7))
            # corpus BLEU paired bootstrap
            rng = np.random.default_rng(1)
            n = len(common)
            idx = rng.integers(0, n, size=(args.n_boot_bleu, n))
            ref_r = [ref_toks[i] for i in range(n)]
            ht = [toks(data[key_t][y]["gen"]) if data[key_t][y]["gen"].strip() else []
                  for y in common]
            hb = [toks(data[key_b][y]["gen"]) if data[key_b][y]["gen"].strip() else []
                  for y in common]
            boot_d = []
            for row in idx:
                bt = corpus_bleu([ref_r[i] for i in row], [ht[i] for i in row])
                bb = corpus_bleu([ref_r[i] for i in row], [hb[i] for i in row])
                boot_d.append(bt - bb)
            entry["bleu4_corpus_delta"] = float(
                summaries[key_t]["bleu4_corpus"] - summaries[key_b]["bleu4_corpus"])
            entry["bleu4_corpus_delta_ci95"] = [
                float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5))]
            # Holm across the tested metrics for this variant
            pads = holm([pp if not math.isnan(pp) else 1.0 for pp in raw_p])
            for m, pa in zip(entry["per_metric"], pads):
                entry["per_metric"][m]["wilcoxon_p_holm"] = pa
            deltas[key_t] = entry

    # ---- write outputs ----
    out = {
        "n_common_clips": len(common),
        "prompt": data[next(iter(files))][common[0]].get("prompt"),
        "checkpoints": {f"{fam}/{v}": data[(fam, v)][common[0]].get("ckpt", "base")
                        for (fam, v) in data},
        "summaries": {f"{fam}/{v}": s for (fam, v), s in summaries.items()},
        "deltas_vs_base": {f"{fam}/{v}": e for (fam, v), e in deltas.items()},
    }
    with open(str(out_prefix) + ".json", "w") as f:
        json.dump(out, f, indent=2)
    perclip_path = out_prefix.parent / "perclip.csv"
    with open(perclip_path, "w") as f:
        f.write("ytid,model,variant,sbleu,meteor,rougeL,bertscore_f1,words\n")
        for r in perclip_rows:
            bs = "" if r["bertscore_f1"] is None else f'{r["bertscore_f1"]:.5f}'
            f.write(f'{r["ytid"]},{r["model"]},{r["variant"]},{r["sbleu"]:.5f},'
                    f'{r["meteor"]:.5f},{r["rougeL"]:.5f},{bs},{r["words"]}\n')

    # ---- markdown summary ----
    md = ["# Probe B (easy): MusicCaps captioning — base vs GRPO", ""]
    md.append(f"- paired clips: **{len(common)}** (MusicCaps `is_audioset_eval=True`, "
              f"downloaded via yt-dlp, 16 kHz mono, 10 s)")
    md.append(f"- prompt: *{out['prompt']}*")
    md.append("- decoding: greedy, num_beams=1, max_new_tokens=256, repetition_penalty=1.0")
    md.append("")
    md.append("## Corpus metrics per model / variant")
    md.append("")
    head = ["model", "variant", "BLEU-4", "METEOR", "ROUGE-L", "BERTScore-F1",
            "words (mean/med)", "%<5w", "%degen", "%hit_max"]
    md.append("| " + " | ".join(head) + " |")
    md.append("|" + "---|" * len(head))
    for (fam, v), s in sorted(summaries.items()):
        bs = "n/a" if s["bertscore_f1_mean"] is None else f'{s["bertscore_f1_mean"]:.4f}'
        md.append(f'| {fam} | {v} | {s["bleu4_corpus"]:.4f} | {s["meteor_mean"]:.4f} | '
                  f'{s["rougeL_mean"]:.4f} | {bs} | {s["words_mean"]:.0f}/{s["words_median"]:.0f} | '
                  f'{s["pct_lt5_words"]:.1f} | {s["pct_degenerate"]:.1f} | '
                  f'{s["pct_hit_max_new_tokens"]:.1f} |')
    md.append("")
    md.append("## Paired deltas vs base (trained - base), same clips")
    md.append("")
    head2 = ["model", "variant", "metric", "mean Δ", "95% CI", "median Δ",
             "Wilcoxon p (Holm)"]
    md.append("| " + " | ".join(head2) + " |")
    md.append("|" + "---|" * len(head2))
    for key_t, e in sorted(deltas.items()):
        fam, v = key_t
        md.append(f'| {fam} | {v} | BLEU-4 (corpus) | {e["bleu4_corpus_delta"]:+.4f} | '
                  f'[{e["bleu4_corpus_delta_ci95"][0]:+.4f}, {e["bleu4_corpus_delta_ci95"][1]:+.4f}] '
                  f'(bootstrap) | n/a | n/a |')
        for m, d in e["per_metric"].items():
            md.append(f'| {fam} | {v} | {m} | {d["mean_delta"]:+.4f} | '
                      f'[{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] | {d["median_delta"]:+.4f} | '
                      f'{d["wilcoxon_p_holm"]:.4g} |')
        md.append(f'| {fam} | {v} | words | {e["words_mean_delta"]:+.1f} | '
                  f'[{e["words_ci95"][0]:+.1f}, {e["words_ci95"][1]:+.1f}] | n/a | n/a |')
    md.append("")
    md.append("Reading guide: a metric drop whose CI excludes 0 (and Holm-corrected "
              "p < 0.05) = GRPO measurably hurt captioning on that metric; %degen / "
              "words columns show whether the drop is a terseness/format-leakage "
              "effect rather than content loss.")
    with open(str(out_prefix) + ".md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nwrote {out_prefix}.json / .md and {perclip_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
