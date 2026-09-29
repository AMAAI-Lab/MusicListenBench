"""Tests of the metrics, the leaderboard scorer and the headline numbers of the paper."""

import json
import os

import pytest

from musiclistenbench import paths
from musiclistenbench.scoring import metrics as M
from musiclistenbench.scoring import score_submission as S

REF = M.load_reference()
PER_ITEM = os.path.join(paths.RESULTS_DIR, "per_item")


def answers_from(slug):
    out = {}
    for suffix in ("", "_perturb"):
        with open(os.path.join(PER_ITEM, f"{slug}{suffix}.jsonl")) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[M.item_id_from_voice(r["voice"])] = r["predicted"]
    return out


def close(x, pct, tol=0.0006):
    return abs(100 * x - pct) <= tol * 100


def test_reference_has_3000_items():
    assert len(REF) == 3000
    kinds = [r["kind"] for r in REF.values()]
    assert (kinds.count("clean"), kinds.count("flip"), kinds.count("stay")) == (1000, 1000, 1000)


def test_parse_item_id():
    assert M.parse_item_id("Q1_melody__test__100__000000__base") == ("Q1_melody__test__100__000000", "clean", None)
    assert M.parse_item_id("Q3_timbre__test__24-73__000012__stay__hiss") == ("Q3_timbre__test__24-73__000012", "stay", "hiss")
    assert M.item_id_from_voice("Q1_melody/Q1_melody__test__100__000000__base.wav") == "Q1_melody__test__100__000000__base"
    with pytest.raises(ValueError):
        M.parse_item_id("not_an_item")


def test_oracle_and_inverse():
    oracle = {i: r["gold"] for i, r in REF.items()}
    s = M.score(oracle, REF)
    assert s["clean"]["All"]["accuracy"] == 1.0 and s["flip_stay"]["All"]["acc_fs"] == 1.0
    inverse = {i: "B" if r["gold"] == "A" else "A" for i, r in REF.items()}
    s = M.score(inverse, REF)
    assert s["clean"]["All"]["accuracy"] == 0.0 and s["flip_stay"]["All"]["acc_fs"] == 0.0


def test_missing_and_invalid_answers_count_as_wrong():
    oracle = {i: r["gold"] for i, r in REF.items()}
    some = list(oracle)[:100]
    for i in some[:50]:
        del oracle[i]
    for i in some[50:]:
        oracle[i] = "C"
    s = M.score(oracle, REF)
    assert s["n_valid_answers"] == 2900 and s["n_missing_or_invalid"] == 100


@pytest.mark.parametrize("letter,clean,rhythm_fs,all_fs,mr,ffr", [("A", 52.8, 47.8, 49.45, 53.0, 48.1),
                                                                   ("B", 47.2, None, None, 47.0, 51.9)])
def test_always_answer_baselines(letter, clean, rhythm_fs, all_fs, mr, ffr):
    s = M.score(M.always_answer(letter, REF), REF)
    assert close(s["clean"]["All"]["accuracy"], clean)
    if rhythm_fs is not None:
        assert close(s["flip_stay"]["Q4_rhythm"]["acc_fs"], rhythm_fs)
        assert close(s["flip_stay"]["All"]["acc_fs"], all_fs)
    assert close(s["flip_stay"]["All"]["miss_rate"], mr) and close(s["flip_stay"]["All"]["false_flip_rate"], ffr)


def test_chance_intervals_of_the_paper():
    for n, lo, hi in [(250, 43.8, 56.2), (500, 45.6, 54.4), (1000, 46.9, 53.1), (2000, 47.8, 52.2)]:
        a, b = M.chance_interval(n)
        assert abs(a - lo) < 0.06 and abs(b - hi) < 0.06


def test_submission_roundtrip(tmp_path):
    oracle = {i: r["gold"] for i, r in REF.items()}
    path = tmp_path / "sub.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"meta": {"model": "oracle", "version": "1", "readout": "generated",
                                     "used_training_split": "none", "date": "2026-09"}}) + "\n")
        for i, a in oracle.items():
            f.write(json.dumps({"item_id": i, "answer": a}) + "\n")
        f.write(json.dumps({"item_id": "Q1_melody__test__100__000000__base", "answer": "B"}) + "\n")   # duplicate
    meta, answers, problems = S.read_submission(str(path))
    assert meta["model"] == "oracle" and len(answers) == 3000
    assert any("duplicate" in p for p in problems)
    assert S.validate(meta, answers, REF) == []
    assert M.score(answers, REF)["clean"]["All"]["accuracy"] == 1.0


def test_validation_reports_problems():
    meta = {"model": "x"}
    problems = S.validate(meta, {"Q1_melody__test__100__000000__base": "C", "made_up_id": "A"}, REF)
    text = " ".join(problems)
    assert "missing 'version'" in text and "unknown item ids" in text and "not 'A' or 'B'" in text
    assert "no answer" in text


# ------------------------------------------------ numbers printed in the paper

PAPER = {   # slug: (clean All, Acc_FS, MR, FFR) in %, from Tables 2 and 3
    "qwen2_audio": (47.2, 50.7, 47.7, 50.9),
    "qwen2_5_omni": (61.2, 61.2, 36.7, 41.0),
    "audio_flamingo3": (52.7, 51.4, 49.6, 47.6),
    "phi4_multimodal": (52.3, 49.1, 53.5, 48.3),
    "gpt_audio_mini": (51.1, 49.6, 50.6, 50.3),
    "gemini_2_5_pro": (79.2, 74.2, 24.7, 26.9),
    "qwen2_5_omni_grpo_lora": (99.0, 86.0, 1.9, 26.1),
    "qwen2_5_omni_grpo_full": (96.2, 83.5, 4.2, 28.9),
    "qwen2_audio_grpo_lora": (68.9, 63.8, 29.7, 42.7),
    "audio_flamingo3_grpo_full": (79.4, 70.4, 22.5, 36.7),
}


@pytest.mark.parametrize("slug", sorted(PAPER))
def test_paper_headline_numbers(slug):
    if not os.path.exists(os.path.join(PER_ITEM, f"{slug}.jsonl")):
        pytest.skip("per-item results not present")
    s = M.score(answers_from(slug), REF)
    clean, acc_fs, mr, ffr = PAPER[slug]
    fs = s["flip_stay"]["All"]
    assert abs(100 * s["clean"]["All"]["accuracy"] - clean) <= 0.051
    assert abs(100 * fs["acc_fs"] - acc_fs) <= 0.051
    assert abs(100 * fs["miss_rate"] - mr) <= 0.051 and abs(100 * fs["false_flip_rate"] - ffr) <= 0.051


def test_two_rate_drop_test_matches_table_16():
    if not os.path.isdir(PER_ITEM):
        pytest.skip("per-item results not present")
    before = M.score(answers_from("qwen2_5_omni"), REF)["flip_stay"]["All"]
    after = M.score(answers_from("qwen2_5_omni_grpo_lora"), REF)["flip_stay"]["All"]
    z, p = M.two_rate_drop_test(before["miss_rate"], after["miss_rate"],
                                before["false_flip_rate"], after["false_flip_rate"])
    assert abs(z - 7.6) < 0.06 and p < 0.001
