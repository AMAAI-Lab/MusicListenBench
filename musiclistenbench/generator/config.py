"""Loads and cross-references the four config YAMLs.

Every other module reads configuration through `load_config()` rather than
parsing YAML itself, so there is exactly one place that resolves paths and
applies row overrides.
"""

import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
SOUNDFONT_DIR = os.environ.get("MLB_SOUNDFONT_DIR", os.path.join(PROJECT_ROOT, "assets", "soundfonts"))


def _load_yaml(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


class Config:
    """Bag of the four parsed configs plus resolved absolute paths."""

    def __init__(self, base, tasks, transforms, splits, output_root=None):
        self.base = base
        self.tasks = tasks["tasks"]
        self.transforms = transforms
        self.splits = splits

        root = output_root or os.environ.get("MLB_OUTPUT_ROOT") or base["paths"]["output_root"]
        self.output_root = root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
        self.audio_dir = os.path.join(self.output_root, base["paths"]["audio_subdir"])
        self.manifest_path = os.path.join(self.output_root, base["paths"]["manifest_name"])
        self.artifacts_dir = os.path.join(self.output_root, base["paths"]["artifacts_subdir"])

        self.sample_rate = base["audio"]["sample_rate"]
        self.target_lufs = base["audio"]["target_lufs"]
        self.max_clip_seconds = base["audio"]["max_clip_seconds"]
        self.tolerances = base["tolerances"]
        self.leak_auc_max = base["leak_audit"]["auc_max"]
        self.seed = base["seed"]

    def active_tasks(self):
        """Task rows that are not excluded (i.e. not stem_surgery)."""
        return {k: v for k, v in self.tasks.items() if v.get("status") != "excluded_failed_certification"}

    def families_for(self, task, split):
        """Nuisance family list for `task`'s `split` side, honouring row_overrides."""
        override = self.transforms.get("row_overrides", {}).get(task)
        if override is not None:
            return list(override[f"{split}_families"])
        return list(self.transforms["split"][f"{split}_families"])

    def soundfonts_for(self, split):
        def resolve(p):   # bare file names live in MLB_SOUNDFONT_DIR (default assets/soundfonts/)
            return p if os.path.isabs(p) else os.path.join(SOUNDFONT_DIR, p)
        return {k: resolve(self.base["soundfonts"][k]) for k in self.base["soundfont_split"][split]}

    def extra_invariance_axes(self, task):
        return self.transforms.get("extra_invariance_axes", {}).get(task, {})


def load_config(output_root=None):
    return Config(
        base=_load_yaml("base.yaml"),
        tasks=_load_yaml("tasks.yaml"),
        transforms=_load_yaml("transforms.yaml"),
        splits=_load_yaml("splits.yaml"),
        output_root=output_root,
    )
