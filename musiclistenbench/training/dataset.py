"""Loads the base-only train set built by
`build_probe.py --split train --json-name train.json` into a flat list of
prompts. No I/O beyond the one JSON read; audio itself is loaded lazily by
the training loop.
"""

import json
import os


def load_train_set(train_json_path, audio_root):
    with open(train_json_path) as f:
        items = json.load(f)
    examples = []
    for item in items:
        voice_relpath = item["voice"][0]
        task = voice_relpath.split("/")[0]
        question = item["conversations"][0]["value"].replace("<audio>", "").strip()
        gold = item["conversations"][1]["value"].strip().upper()
        examples.append({
            "audio_path": os.path.join(audio_root, voice_relpath),
            "task": task,
            "question": question,
            "gold": gold,
        })
    return examples


class EpochSampler:
    """Shuffled, without-replacement sampling within an epoch; reshuffles and
    starts a new epoch once exhausted. Deterministic under a fixed seed."""

    def __init__(self, examples, rng):
        self.examples = examples
        self.rng = rng
        self._order = []

    def _refill(self):
        self._order = list(range(len(self.examples)))
        self.rng.shuffle(self._order)

    def sample(self, k):
        batch = []
        while len(batch) < k:
            if not self._order:
                self._refill()
            take = min(k - len(batch), len(self._order))
            idxs, self._order = self._order[:take], self._order[take:]
            batch.extend(self.examples[i] for i in idxs)
        return batch
