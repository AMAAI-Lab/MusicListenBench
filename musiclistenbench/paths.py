"""Default locations shared by every entry point.

All defaults can be overridden on the command line. Two environment variables
move whole directories without touching any flag:

    MLB_DATA_DIR      folder holding train.json, eval.json, eval_perturb.json,
                      eval_perturb.meta.jsonl and the audio folders
                      (audio/ and audio_perturb/)
    MLB_RESULTS_DIR   folder that receives per-item result files
"""

import gzip
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PKG_DIR)

DATA_DIR = os.environ.get("MLB_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RESULTS_DIR = os.environ.get("MLB_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
GENERATED_DIR = os.path.join(REPO_ROOT, "generated")

EVAL_JSON = os.path.join(DATA_DIR, "eval.json")
EVAL_PERTURB_JSON = os.path.join(DATA_DIR, "eval_perturb.json")
EVAL_PERTURB_META = os.path.join(DATA_DIR, "eval_perturb.meta.jsonl")
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
AUDIO_PERTURB_DIR = os.path.join(DATA_DIR, "audio_perturb")
MANIFEST = os.path.join(DATA_DIR, "manifest.jsonl.gz")


def open_text(path, mode="rt"):
    """`open` that also reads/writes gzip files (used for the item manifest)."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def resolve_clip_path(path, clip_root):
    """Manifest rows store clip paths relative to the generator output folder
    (audio/<item>/<clip>.wav); absolute paths are left unchanged."""
    return path if os.path.isabs(path) else os.path.join(clip_root, path)
