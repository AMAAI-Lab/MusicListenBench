<h1 align="center">MusicListenBench</h1>

<p align="center"><strong>Can Audio LLMs Hear the Difference Yet?</strong></p>
<p align="center">A paired listening benchmark for melody, harmony, timbre, and rhythm.</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.29375">Paper</a> ·
  <a href="https://huggingface.co/datasets/amaai-lab/MusicListenBench">Dataset</a> ·
  <a href="https://amaai-lab.github.io/MusicListenBench/">Leaderboard</a> ·
  <a href="https://huggingface.co/spaces/amaai-lab/MusicListenBench-leaderboard">Hugging Face Space</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#citation">Citation</a>
</p>

## Benchmark at a glance

Each item joins two clips with a one-second gap. The model answers a single
A/B question about the melody, chord, instrument, or relative tempo.

| Variant | What changes | What it tests |
| --- | --- | --- |
| **FLIP** | The queried musical attribute | Whether the model notices the musical change |
| **STAY** | The sound, through edits such as EQ, hiss, reverb, or transposition | Whether the model preserves its judgment when the queried attribute stays the same |

The leaderboard reports clean accuracy, miss rate on FLIP items, false-flip
rate on STAY items, and FLIP/STAY accuracy. For rates expressed as fractions,
`Acc_FS = 1 - (MR + FFR) / 2`. A model that ignores the audio scores about 50%
on FLIP/STAY.

| Split | Items |
| --- | --- |
| Rendered training | 10,000 |
| Rendered test | 1,000 clean + 2,000 FLIP/STAY |
| Real-recordings test | 600 clean + 1,200 FLIP/STAY |

## Quick start

### Score the included example

Clone the release and run the scorer from the repository root. The example
uses the bundled item files and scores all 3,000 rendered test items.

```bash
git clone --branch release --single-branch https://github.com/AMAAI-Lab/MusicListenBench.git
cd MusicListenBench
python scripts/verify_data.py
python -m musiclistenbench.scoring.score_submission data/example_submission.jsonl
```

### Evaluate a model

1. Install the evaluation dependencies.

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Download the audio from the [Hugging Face dataset](https://huggingface.co/datasets/amaai-lab/MusicListenBench).
   Put the rendered clean test WAVs in `data/audio/` and the rendered
   FLIP/STAY WAVs in `data/audio_perturb/`, keeping the per-task folders in
   both. For separately packaged archives, use [fetch_audio.sh](scripts/fetch_audio.sh)
   with the directory URL that contains the archives and `SHA256SUMS`.

   ```bash
   python scripts/verify_data.py --audio --test-only
   ```

3. Run the clean and FLIP/STAY evaluations. This example uses Qwen2.5-Omni and
   one process on each of four GPUs. Adjust the GPU IDs to match your machine.

   ```bash
   bash scripts/run_probe_gpus.sh 0,1,2,3 --backend qwen2.5-omni --out-name per_item/my_run
   bash scripts/run_probe_gpus.sh 0,1,2,3 --backend qwen2.5-omni --perturb --out-name per_item/my_run_perturb
   ```

4. Convert the predictions to the [submission format](musiclistenbench/scoring/score_submission.py)
   and score the result.

   ```bash
   python -m musiclistenbench.scoring.convert_results \
       --clean results/per_item/my_run.jsonl \
       --flip-stay results/per_item/my_run_perturb.jsonl \
       --model "My model" --version v1 --readout logprob \
       --used-training-split none --date "$(date +%F)" --out submission.jsonl
   python -m musiclistenbench.scoring.score_submission submission.jsonl
   ```

See [Submit a result](https://amaai-lab.github.io/MusicListenBench/submit.html) for the leaderboard CSV
format and pull request instructions.

## Repository guide

| Location | Contents |
| --- | --- |
| [data/](data/) | Item files, the generator manifest, and a scored submission example |
| [musiclistenbench/generator/](musiclistenbench/generator/) and [benchmark/](musiclistenbench/benchmark/) | Seeded symbolic music generation with music21 and FluidSynth, plus the paired audio and variants |
| [musiclistenbench/eval/](musiclistenbench/eval/) and [backends/](musiclistenbench/backends/) | Evaluation runners for eight open and four commercial audio models |
| [musiclistenbench/scoring/](musiclistenbench/scoring/) | Submission scoring, metrics, and scripts for the paper's tables |
| [musiclistenbench/training/](musiclistenbench/training/), [transfer/](musiclistenbench/transfer/), and [captioning/](musiclistenbench/captioning/) | GRPO training, leave-one-task-out runs, and captioning evaluation |
| [musiclistenbench/other_sets/](musiclistenbench/other_sets/) | Real-recordings and Slakh2100 builders. Set `REAL_DATA_DIR` or `SLAKH_DIR` to the source datasets. |
| [scripts/](scripts/) and [tests/](tests/) | Audio and data checks, [Slakh training](scripts/train_slakh.sh), [real-recordings or Slakh evaluation](scripts/eval_other_set.sh), and scoring tests |
| [results/captioning/](results/captioning/) | IDs and scores for 2,735 MusicCaps clips |

Bit-exact reproduction of the rendered audio also depends on the FluidSynth
and ffmpeg versions.

## Citation

If you use MusicListenBench, please cite [the paper](https://arxiv.org/abs/2611.29375).

```bibtex
@misc{roy2026musiclistenbench,
  title = {{MusicListenBench}: Can Audio {LLM}s Hear the Difference Yet?},
  author = {Roy, Abhinaba and Mehrish, Ambuj and Herremans, Dorien},
  year = {2026},
  eprint = {2611.29375},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD},
  url = {https://arxiv.org/abs/2611.29375}
}
```

## Licences

The code is free to use, copy, and modify. Follow the licences for FluidR3 GM,
GeneralUser GS, and MuseScore General when using the rendered audio. The
real-recordings test set follows the source dataset terms listed on the
[dataset card](https://huggingface.co/datasets/amaai-lab/MusicListenBench#licences). The MusicCaps results include
clip IDs and scores.
