"""Base-only reward: a parse term and a correctness term only. Terms that would
reward invariance to sound-only changes or sensitivity to content edits are
deliberately absent: they need re-rendered variants, and GRPO here trains on
clean base pairs only (Appendix D of the paper).

Pure function, no I/O, no model calls -- unit-testable on its own.
"""

import re

PARSE_WEIGHT = 0.2
CORRECT_WEIGHT = 1.0


def parse_letter(text):
    match = re.search(r"[AaBb]", text)
    return match.group(0).upper() if match else None


def compute_reward(raw_text, gold):
    """Returns (reward, parsed_letter, correct)."""
    parsed = parse_letter(raw_text)
    parses = parsed is not None
    correct = parsed == gold
    reward = PARSE_WEIGHT * float(parses) + CORRECT_WEIGHT * float(correct)
    return reward, parsed, correct


def balance_adjustment(predicted, skew, coef=0.1):
    """Anti-collapse nudge, added on top of `compute_reward`'s output for a
    single rollout. `skew` is (recent fraction of this task's rollouts that
    predicted 'B') - 0.5, tracked by the caller (see `PredLetterBalance` in
    train_grpo.py) -- skew > 0 means the policy currently over-predicts 'B'
    for this task, skew < 0 means it over-predicts 'A'.

    This file has no idea what 'A'/'B' mean for a given task (same/
    different, first/second, or anything else) -- it only ever reacts to
    whichever letter is CURRENTLY over-represented in the policy's own
    recent outputs, so the same call handles a same/different collapse on
    one task and a first/second collapse on another without any per-task
    special-casing. It discourages the over-represented letter and rewards
    the under-represented one, scaled by how skewed the recent policy is.

    Unlike scaling `compute_reward`'s correctness term by a per-class
    weight, this must be applied per rollout (varying with THIS rollout's
    own predicted letter), not per group -- GRPO normalises reward within
    each group of G rollouts sharing one prompt/gold letter, and that
    normalisation is invariant to any adjustment that is constant across
    the whole group (e.g. a single weight applied uniformly because the
    group's gold letter is currently the hard class). A per-rollout term
    that depends on what THIS rollout actually said survives that
    normalisation, because different rollouts in the same group can and do
    sample different letters under temperature>0 generation.
    """
    if predicted == "B":
        return -coef * skew
    if predicted == "A":
        return coef * skew
    return 0.0
