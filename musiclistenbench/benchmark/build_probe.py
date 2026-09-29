"""Builds a standalone, SonicBench-style (arXiv 2601.11039) probe set for the
four base questions (Q1_melody, Q2_harmony, Q3_timbre, Q4_rhythm).

SonicBench's `probe_json/*/eval.json` tests one specific presentation
strategy: concatenate a trial's two clips into ONE physical audio file with a
short silence between them, then ask a single letter ('A'/'B') question about
it, with many paraphrased wordings of the same question. This script applies
that exact strategy to our own base (non-catch, non-invariance,
non-equivariance) trials, so we can check whether it elicits sane behaviour
here too. Both clips go into ONE audio file, so a model that accepts only a
single audio input can be tested exactly like one that accepts several.

Output layout mirrors probe_json's own convention:
  data/audio/<task>/<trial_id>.wav   -- the combined clip
  data/eval.json                     -- [{"voice": [...], "conversations": [...]}]

Run: python -m musiclistenbench.benchmark.build_probe
"""

import argparse
import json
import os
import random

import numpy as np
import soundfile as sf

from musiclistenbench import paths

TASKS = ["Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"]

TAIL_SAME_DIFFERENT = (
    "Only answer 'A' for the same, 'B' for different. "
    "Do not add any explanation, punctuation, or extra text. <audio>"
)

# Flipped mapping, used under --counterbalance-letters so the letter carries
# no information about the answer -- Q1-Q3's A=same/B=different pinning is
# an exploitable shortcut on its own (a policy can just learn "prefer this
# token" instead of listening), independent of the audio-level byte-identity
# shortcut. Q4_rhythm doesn't need this: its gold letter already depends on
# both which physical clip is faster and which polarity template got picked,
# so it isn't pinned to a fixed semantic meaning the way Q1-Q3 are.
TAIL_SAME_DIFFERENT_FLIPPED = (
    "Only answer 'B' for the same, 'A' for different. "
    "Do not add any explanation, punctuation, or extra text. <audio>"
)

# Q1_melody / Q2_harmony / Q3_timbre: identity question, no polarity flip
# needed (A=same, B=different is fixed across every paraphrase).
TEMPLATES_SAME_DIFFERENT = {
    "Q1_melody": [
        "Two melodic clips are joined by a brief silence. Do they play the same melody, or a different one?",
        "You will hear two short melodies separated by a short pause. Is the melody the same in both, or different?",
        "Listen to two melodic phrases with a brief gap between them. Same tune, or different?",
        "A pair of melodies is presented back to back with a short silence in between. Are they the same melody?",
        "Two musical phrases follow one another after a brief pause. Is the second phrase the same melody as the first, or different?",
        "You are given two clips of melodic material with a short silent gap. Do they share the same melody?",
        "Listen carefully to the two melodic clips, divided by a brief silence. Same melody, or not?",
        "Two short tunes are played in sequence with a pause between them. Is it the same melody twice, or two different melodies?",
    ],
    "Q2_harmony": [
        "Two chords are played in sequence with a brief silence between them. Is the chord the same in both, or different?",
        "You will hear two chords separated by a short pause. Same chord, or different?",
        "Listen to two chords joined by a brief gap of silence. Do they sound like the same kind of chord, or different?",
        "A pair of chords is presented back to back with a short silence in between. Is the harmony the same, or different?",
        "Two short chords follow each other after a brief pause. Same chord type, or different?",
        "You are given two chord clips with a short silent gap. Do they share the same chord type?",
        "Listen to the two chords, divided by a brief silence. Same harmonic quality, or not?",
        "Two chords are played one after the other with a pause between them. Is it the same chord twice, or two different qualities?",
    ],
    "Q3_timbre": [
        "Two clips are joined by a brief silence. Are they played by the same instrument, or different instruments?",
        "You will hear two short clips separated by a short pause. Same instrument in both, or different?",
        "Listen to two clips with a brief gap between them. Is it the same instrument, or a different one?",
        "A pair of clips is presented back to back with a short silence in between. Are they performed on the same instrument?",
        "Two clips follow one another after a brief pause. Is the second clip the same instrument as the first, or different?",
        "You are given two clips with a short silent gap. Do they share the same instrument?",
        "Listen carefully to the two clips, divided by a brief silence. Same instrument, or not?",
        "Two clips are played in sequence with a pause between them. Is it the same instrument twice, or two different instruments?",
    ],
}

# Q4_rhythm: genuine comparative (first vs second), so polarity ("faster" vs
# "slower") flips which letter is gold -- same trick SonicBench's own
# tempo_comparison split uses. Each entry is (question_text, polarity).
TEMPLATES_RHYTHM = [
    ("Two audio segments are given, with a brief silence in between. Which segment moves faster?", "faster"),
    ("Two audio segments are given, with a brief silence in between. Which segment moves slower?", "slower"),
    ("You are presented with two sound clips, separated by a short silent interval. Which clip carries more speed?", "faster"),
    ("You are presented with two sound clips, separated by a short silent interval. Which part proceeds at a slower pace?", "slower"),
    ("Listen to a pair of audio segments with a brief silent space separating them. Which audio feels slower?", "slower"),
    ("You are given two clips with a small silent gap in between. Which segment moves faster?", "faster"),
    ("Two auditory recordings are given, separated by a small pause. Which part has a faster rhythm?", "faster"),
    ("You are provided with two audio clips separated by a short silent gap. Which part proceeds at a faster pace?", "faster"),
    ("Two sound recordings are presented, separated by a brief silent segment. Which clip runs more leisurely?", "slower"),
    ("You have two auditory segments, separated by a short pause. Which segment plays more rapidly?", "faster"),
]
TAIL_RHYTHM = (
    "Only answer 'A' for the first, 'B' for the second. "
    "Do not add any explanation, punctuation, or extra text. <audio>"
)


def load_manifest(path, clip_root=None):
    """Reads the item manifest (plain or .gz). Relative clip paths are resolved
    against `clip_root` (default: the folder holding the manifest)."""
    rows = []
    clip_root = clip_root or os.path.dirname(os.path.abspath(path))
    with paths.open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task" not in row:  # header line
                continue
            row["clip_paths"] = [paths.resolve_clip_path(p, clip_root) for p in row["clip_paths"]]
            rows.append(row)
    return rows


def select_base_rows(rows, split):
    selected = {t: [] for t in TASKS}
    for row in rows:
        if row["task"] not in TASKS:
            continue
        if row["expected_role"] != "base" or row["split"] != split or row["is_catch"]:
            continue
        selected[row["task"]].append(row)
    for t in TASKS:
        selected[t].sort(key=lambda r: r["trial_id"])
    return selected


def build_combined_clip(clip_paths, silence_seconds, sample_rate):
    a, sr_a = sf.read(clip_paths[0], dtype="float32", always_2d=False)
    b, sr_b = sf.read(clip_paths[1], dtype="float32", always_2d=False)
    assert sr_a == sample_rate and sr_b == sample_rate, (
        f"expected {sample_rate} Hz source clips, got {sr_a}/{sr_b}"
    )
    silence = np.zeros(int(round(silence_seconds * sample_rate)), dtype=np.float32)
    return np.concatenate([a, silence, b])


def make_entry(task, row, rng, voice_relpath, counterbalance=False):
    if task == "Q4_rhythm":
        question, polarity = rng.choice(TEMPLATES_RHYTHM)
        prompt = f"{question} {TAIL_RHYTHM}"
        faster_letter = "A" if row["answer"] == "first" else "B"
        slower_letter = "B" if faster_letter == "A" else "A"
        gold = faster_letter if polarity == "faster" else slower_letter
    else:
        question = rng.choice(TEMPLATES_SAME_DIFFERENT[task])
        flip = counterbalance and rng.random() < 0.5
        tail = TAIL_SAME_DIFFERENT_FLIPPED if flip else TAIL_SAME_DIFFERENT
        prompt = f"{question} {tail}"
        same_letter, diff_letter = ("B", "A") if flip else ("A", "B")
        gold = same_letter if row["answer"] == "same" else diff_letter
    return {
        "voice": [voice_relpath],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": gold},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=os.path.join(paths.GENERATED_DIR, "manifest.jsonl"),
                        help="item manifest written by the generator (plain or .gz)")
    parser.add_argument("--clip-root", default=None,
                        help="folder that relative clip paths in the manifest are relative to "
                             "(default: the manifest's own folder)")
    parser.add_argument("--out-dir", default=paths.DATA_DIR)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--silence-seconds", type=float, default=1.0,
                         help="silence between clip A and clip B (1.0 s in the paper)")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-per-task", type=int, default=None,
                         help="cap items per task, for a quick smoke run")
    parser.add_argument("--json-name", default="eval.json",
                         help="output filename, e.g. train.json when --split train")
    parser.add_argument("--counterbalance-letters", action="store_true",
                         help="Q1-Q3 only: randomize per-item whether 'A' or 'B' means "
                              "'same', so the letter itself carries no information about "
                              "the answer. Use for train.json; leave eval.json fixed-mapping "
                              "so accuracy stays comparable across runs.")
    args = parser.parse_args()

    rows = load_manifest(args.manifest, args.clip_root)
    by_task = select_base_rows(rows, args.split)

    audio_root = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_root, exist_ok=True)

    entries = []
    rng = random.Random(args.seed)
    for task in TASKS:
        task_rows = by_task[task]
        if args.limit_per_task is not None:
            task_rows = task_rows[: args.limit_per_task]
        task_dir = os.path.join(audio_root, task)
        os.makedirs(task_dir, exist_ok=True)
        for row in task_rows:
            clip = build_combined_clip(row["clip_paths"], args.silence_seconds, args.sample_rate)
            out_name = f"{row['trial_id']}.wav"
            sf.write(os.path.join(task_dir, out_name), clip, args.sample_rate, subtype="PCM_16")
            voice_relpath = f"{task}/{out_name}"
            entries.append(make_entry(task, row, rng, voice_relpath, counterbalance=args.counterbalance_letters))
        print(f"{task}: {len(task_rows)} items")

    eval_path = os.path.join(args.out_dir, args.json_name)
    with open(eval_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"wrote {len(entries)} items -> {eval_path}")
    print(f"wrote combined audio -> {audio_root}/<task>/<trial_id>.wav")


if __name__ == "__main__":
    main()
