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

## Anonymous reviewer copy

`scripts/build.py` also builds `_anon/`: the same site with the lab name,
GitHub, dataset, paper and contact links removed, the Submit page and BibTeX
replaced by a note, identifying CSV fields blanked, and search engines blocked.
It is deployed to Cloudflare Pages at `https://<CF_PROJECT>.pages.dev`, whose
URL does not reveal the account owner.

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
