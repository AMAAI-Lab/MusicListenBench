# MusicListenBench leaderboard

Can audio LLMs hear the difference yet? This repo holds the leaderboard for
MusicListenBench. The dataset lives at
https://huggingface.co/datasets/amaai-lab/MusicListenBench

## How it works

`leaderboard.csv` is the single source of truth. On every push to `main`, a
GitHub Action validates it and publishes the same static site (`site/`) with
the CSV to:

- GitHub Pages
- a static Hugging Face Space (`space/README.md` holds the Space config)

Nobody edits the Space directly.

## Adding a result

Open a pull request that adds rows to `leaderboard.csv`. The `Validate
leaderboard` check runs `scripts/validate.py` and reports problems inline.
See `site/submit.html` for the contributor instructions.

| column | meaning |
|---|---|
| type | `model`, or `baseline` for reference lines like chance (not ranked) |
| access | `open` or `closed` weights |
| params_b | parameters in billions, empty if unknown |
| method | e.g. `zero-shot`, `SFT`, `GRPO` |
| melody…harmony | percent accuracy, 500 items each |
| overall | mean of the four aspects (TODO: confirm) |
| verified | `yes` once maintainers reproduce the result |
| benchmark_version | `1.0` |

## One-time setup

1. Settings → Pages → Source: **GitHub Actions**.
2. Create the Space `amaai-lab/MusicListenBench-leaderboard` (SDK: static), or
   change `HF_SPACE` in `.github/workflows/deploy.yml`.
3. Add a Hugging Face write token as the repo secret `HF_TOKEN`. Without it,
   only GitHub Pages deploys.
4. Fill the placeholders in `site/assets/config.js` and search the site for
   `TODO`.

## Local preview

```
cp leaderboard.csv site/ && cd site && python -m http.server
```
