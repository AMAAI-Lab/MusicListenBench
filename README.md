# MusicListenBench leaderboard

MusicListenBench tests how well audio language models hear differences in
melody, harmony, timbre, and rhythm. This repository maintains the public leaderboard.

[Paper](https://arxiv.org/abs/2611.29375) · [Dataset](https://huggingface.co/datasets/amaai-lab/MusicListenBench) · [Evaluation code](https://github.com/AMAAI-Lab/MusicListenBench/tree/release) · [Leaderboard](https://amaai-lab.github.io/MusicListenBench/) · [Hugging Face Space](https://huggingface.co/spaces/amaai-lab/MusicListenBench-leaderboard)

## How it works

`leaderboard.csv` is the single source of truth. On every push to `main`, a
GitHub Action validates it and publishes the same static site (`site/`) with
the CSV to:

- GitHub Pages
- a static Hugging Face Space (`space/README.md` holds the Space config)

Update the Space card in [space/README.md](space/README.md). GitHub Actions
publishes it with the leaderboard.

## Adding a result

Open a pull request that adds rows to `leaderboard.csv`. The `Validate
leaderboard` check runs `scripts/validate.py` and reports problems inline.
See `site/submit.html` for the contributor instructions.

| column | meaning |
|---|---|
| type | `model`, or `baseline` for reference lines like chance (not ranked) |
| access | `open` or `closed` weights |
| domain | `music` for models built for music understanding (own section), `general` for other audio LLMs |
| params_b | parameters in billions, empty if unknown |
| method | e.g. `zero-shot`, `SFT`, `GRPO` |
| melody…harmony | clean accuracy by task |
| overall | mean clean accuracy across the four tasks |
| verified | `yes` once maintainers reproduce the result |
| benchmark_version | `1.0` |

## Anonymous reviewer copy

`scripts/build.py` also builds `_anon/`: the same site with the lab name,
GitHub, dataset, paper and contact links removed, the Submit page and BibTeX
replaced by a note, identifying CSV fields blanked, and search engines blocked.
To publish the reviewer copy, run the deployment workflow manually with
`deploy_reviewer` enabled.

The build **fails** if any term in `LEAK_TERMS` (in `scripts/build.py`) appears
anywhere in `_anon/`. Add all author surnames there before submitting.
Set anonymous links in `site/assets/config.anon.js`.

Preview locally: `python scripts/build.py && cd _anon && python -m http.server`

After decisions: delete the Cloudflare project and remove its secrets.

## One-time setup

1. Settings → Pages → Source: **GitHub Actions**.
2. Create the Space `amaai-lab/MusicListenBench-leaderboard` (SDK: static), or
   change `HF_SPACE` in `.github/workflows/deploy.yml`.
3. Add a Hugging Face write token as the repo secret `HF_TOKEN`. Without it,
   only GitHub Pages deploys.
4. For the anonymous copy, add repo secrets `CLOUDFLARE_API_TOKEN` (a token
   with the "Cloudflare Pages: Edit" permission) and `CLOUDFLARE_ACCOUNT_ID`.
5. Fill the placeholders in `site/assets/config.js` and search the site for
   `TODO`.

## Local preview

```
python scripts/build.py && cd _site && python -m http.server
```

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
