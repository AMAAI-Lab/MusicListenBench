# MusicListenBench

Code and item files for the paper *MusicListenBench: Can Audio LLMs Hear the Difference Yet?* (under
double-blind review; authors anonymous). The leaderboard is at https://musiclistenbench-review.pages.dev .

Each test item is one audio file (clip A, one second of silence, clip B) and one question about the pair:
same melody, same chord, same instrument, or which clip is faster. Every item has two variants. In the FLIP
variant the music changes, so the correct answer changes. In the STAY variant only the sound changes (room
echo, EQ, background hiss, transposition), so the correct answer stays. A model that ignores the audio
therefore scores 50% on average, whatever letter it prefers. The miss rate (MR) is the share of FLIP items
answered wrongly, the false-flip rate (FFR) the share of STAY items answered wrongly, and the FLIP/STAY
accuracy is `Acc_FS = 1 - (MR + FFR) / 2`.

The training split has 10,000 items (2,500 per task). The test split has 1,000 clean items (250 per task) and
2,000 FLIP/STAY items.

## Layout

* `data/`: item files (`train.json`, `eval.json`, `eval_perturb.json`, `eval_perturb.meta.jsonl`), the generator
  manifest `manifest.jsonl.gz`, and `example_submission.jsonl`. The audio is not stored here.
* `musiclistenbench/generator/`: renders the audio from symbolic music with music21 and FluidSynth.
* `musiclistenbench/benchmark/`: builds the single-file test items and the FLIP/STAY files.
* `musiclistenbench/scoring/`: the scoring script, the leaderboard scorer and the script that recomputes the
  paper's tables.
* `musiclistenbench/eval/`, `musiclistenbench/backends/`: evaluation code for eight open and four commercial
  audio LLMs.
* `musiclistenbench/training/`: GRPO training (LoRA and full fine-tuning).
* `musiclistenbench/transfer/`, `musiclistenbench/captioning/`: leave-one-task-out test and the captioning check.
* `results/captioning/`: the 2,735 MusicCaps clip ids of the captioning check and the per-clip scores.
* `scripts/`: runners, audio download, data check. `tests/`: tests for the scoring code.

## Quick start

Run everything from the repository root.

```bash
pip install -r requirements.txt

python scripts/verify_data.py                       # item files against the numbers in the paper

bash scripts/fetch_audio.sh <archive-url> [--test-only]   # or regenerate the audio with the generator
python scripts/verify_data.py --audio

# evaluate a model (letter probabilities, one process per GPU)
bash scripts/run_probe_gpus.sh 0,1,2,3 --backend qwen2.5-omni --out-name per_item/my_run
bash scripts/run_probe_gpus.sh 0,1,2,3 --backend qwen2.5-omni --perturb --out-name per_item/my_run_perturb

# turn the two result files into a leaderboard submission and score it
python -m musiclistenbench.scoring.convert_results --clean results/per_item/my_run.jsonl \
    --flip-stay results/per_item/my_run_perturb.jsonl --model "My model" --version v1 \
    --readout logprob --used-training-split none --date 2026-09-30 --out submission.jsonl
python -m musiclistenbench.scoring.score_submission submission.jsonl
```

The submission format is described at the top of `musiclistenbench/scoring/score_submission.py`.
Leaderboard rows are added to `leaderboard.csv`.

The audio is generated from a seeded generator, so the items are reproducible; bit-exact audio also depends
on the FluidSynth and ffmpeg versions.

## Licences

The code is free to use, copy and modify.

The audio is rendered with the soundfonts FluidR3 GM, GeneralUser GS and MuseScore General; follow the licence
terms that come with them. The benchmark audio contains no recordings of people and no copyrighted musical
works. MusicCaps clips are not redistributed.
