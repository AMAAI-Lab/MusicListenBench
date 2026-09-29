"""Download the MusicCaps *test set* (is_audioset_eval == True) audio and build
a frozen manifest for the captioning probe (Probe B, easy version).

Notes
-----
* MusicCaps has no official train/test split column other than `is_audioset_eval`
  (2,858 of 5,521 rows). Verified locally: the balanced subset (1,000) is fully
  contained in it, so "the test set" == is_audioset_eval == True.
* Audio comes from YouTube via yt-dlp (pinned to whatever version is installed
  -- its version is logged into the manifest). Each clip is cut to
  exactly [start_s, end_s) (10 s), re-encoded to 16 kHz mono PCM_16 wav, sha256'd.
  `--force-keyframes-at-cuts` is required: with stream-copy cuts the extracted
  section overshoots to keyframe boundaries (a "*30-40" request yields ~20 s).
* Resumable: existing valid wavs already listed in the manifest are skipped;
  previously failed ids are retried (availability changes over time).

Run (from the repository root; clips land in ./musiccaps/):
    python -m musiclistenbench.captioning.download_musiccaps --target 2858
"""

import argparse
import ast
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import urllib.request
from concurrent import futures as cf
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from musiclistenbench import paths

DATA = Path(paths.REPO_ROOT) / "musiccaps"
AUDIO = DATA / "audio"
CSV_PATH = DATA / "musiccaps-public.csv"
MANIFEST = DATA / "manifest.jsonl"
FAILED_LOG = DATA / "failed.tsv"

CSV_URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"
YTDLP = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")


def _ensure_node_on_path() -> None:
    # Recent yt-dlp needs a JavaScript runtime (node or deno) to pass YouTube's
    # "n" challenge. Install one and put it on PATH before running this script.
    if not shutil.which("node"):
        print("warning: `node` not found on PATH; yt-dlp may fail to download clips")


_ensure_node_on_path()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_csv() -> None:
    if CSV_PATH.exists():
        return
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {CSV_URL} -> {CSV_PATH}")
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)


def eval_rows(seed: int):
    with open(CSV_PATH, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["is_audioset_eval"] == "True"]
    rng = random.Random(seed)  # deterministic order so retries target the same ids
    rng.shuffle(rows)
    return rows


def fetch_one(row: dict, timeout: float, cookies=None, cookies_from_browser=None):
    import soundfile as sf

    ytid = row["ytid"]
    start, end = float(row["start_s"]), float(row["end_s"])
    wav = AUDIO / f"{ytid}.wav"
    cmd = [
        YTDLP,
        "--js-runtimes", "node", "--remote-components", "ejs:github",
        "-f", "bestaudio/best",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(AUDIO / f"{ytid}.%(ext)s"),
        "--no-playlist", "--retries", "2", "--fragment-retries", "2",
        "--sleep-requests", "0.2",
        "-q", "--no-warnings",
    ]
    # cmd = [
    #     YTDLP,
    #     "--js-runtimes", "node",
    #     "--remote-components", "ejs:github",
    #     "-f", "bestaudio/best",
    #     "--download-sections", f"*{start}-{end}",
    #     "--force-keyframes-at-cuts",
    #     "-x", "--audio-format", "wav",
    #     "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
    #     "-o", str(AUDIO / f"{ytid}.%(ext)s"),
    #     "--no-playlist",
    #     "--min-sleep-interval", "3",
    #     "--max-sleep-interval", "7",
    #     # "--cookies-from-browser", "chrome",  # Change to "firefox", "edge", or "safari" if needed
    #     "--extractor-args", "youtube:player-client=ios,web_safari",
    #     "--retries", "10",
    #     "--fragment-retries", "10",
    #     "--file-access-retries", "5",

    # ]
    # YouTube's IP-level "confirm you're not a bot" flag (yt-dlp#11868/#12045):
    # pass real logged-in cookies to ride out the block. Netscape-format file
    # exported from a LOGGED-IN browser profile (incognito exports do NOT work).
    if cookies:
        cmd += ["--cookies", str(cookies)]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd.append(f"https://www.youtube.com/watch?v={ytid}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0 or not wav.exists():
        return None, ((r.stderr or "").strip().splitlines() or ["unknown"])[-1][:250]

    # validate + trim to exactly (end - start) seconds, rewrite as PCM_16
    try:
        y, sr = sf.read(wav, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        want = end - start
        if len(y) < (want - 0.5) * sr:
            return None, f"too_short_{len(y) / sr:.2f}s"
        y = y[: int(round(want * sr))]
        sf.write(wav, y, sr, subtype="PCM_16")
    except Exception as e:  # unreadable audio
        return None, f"read_error: {e}"

    rec = {
        "ytid": ytid,
        "start_s": start,
        "end_s": end,
        "caption": row["caption"],
        "aspect_list": ast.literal_eval(row["aspect_list"]),
        "audioset_positive_labels": row["audioset_positive_labels"],
        "is_balanced_subset": row["is_balanced_subset"] == "True",
        "wav": f"audio/{ytid}.wav",
        "duration_s": round(len(y) / sr, 3),
        "sha256": sha256_file(wav),
    }
    return rec, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=2858, help="stop when this many clips are in the manifest")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=150.0, help="per-clip yt-dlp timeout (s)")
    ap.add_argument("--max-attempt-factor", type=float, default=2.5,
                    help="give up after target*factor download attempts this run")
    ap.add_argument("--cookies", default=None,
                    help="Netscape cookies.txt from a LOGGED-IN YouTube browser profile "
                         "(works around the 'not a bot' IP flag, yt-dlp#12045)")
    ap.add_argument("--cookies-from-browser", dest="cookies_from_browser", default=None,
                    help="e.g. chrome, firefox, edge:chromium ... (reads the logged-in profile)")
    args = ap.parse_args()

    AUDIO.mkdir(parents=True, exist_ok=True)
    ensure_csv()

    done = {}
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["ytid"]] = rec
    print(f"resuming: {len(done)} clips already in {MANIFEST}")

    ytdlp_ver = subprocess.run([YTDLP, "--version"], capture_output=True, text=True).stdout.strip()
    rows = [r for r in eval_rows(args.seed) if r["ytid"] not in done]
    print(f"eval split: candidates remaining {len(rows)} (yt-dlp {ytdlp_ver})")

    n_target = args.target - len(done)
    if n_target <= 0:
        print("target already reached")
        return

    max_attempts = max(50, int(n_target * args.max_attempt_factor))
    rows_iter = iter(rows)
    got, attempted, failures = 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
            open(MANIFEST, "a") as mf, open(FAILED_LOG, "a") as ff:
        pending = {}

        def top_up():
            nonlocal attempted
            while len(pending) < args.workers * 2 and attempted < max_attempts:
                row = next(rows_iter, None)
                if row is None:
                    return
                pending[ex.submit(fetch_one, row, args.timeout,
                                  cookies=args.cookies,
                                  cookies_from_browser=args.cookies_from_browser)] = row
                attempted += 1

        top_up()
        while pending and got < n_target:
            done_futs, _ = cf.wait(set(pending), return_when=cf.FIRST_COMPLETED)
            for fut in done_futs:
                row = pending.pop(fut)
                rec, err = fut.result()
                if rec is None:
                    failures.append((row["ytid"], err))
                    ff.write(f"{row['ytid']}\t{err}\n")
                    ff.flush()
                else:
                    mf.write(json.dumps(rec) + "\n")
                    mf.flush()
                    got += 1
                    if got % 25 == 0:
                        print(f"[{got}/{n_target}] latest: {rec['ytid']}", flush=True)
            top_up()

    print(f"done: +{got} new clips (total {len(done) + got}), {len(failures)} failures this run "
          f"(logged to {FAILED_LOG})")


if __name__ == "__main__":
    main()
