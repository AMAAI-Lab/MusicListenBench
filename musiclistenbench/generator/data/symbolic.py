"""Content construction: music21/MIDI-domain (note list) decisions only.
Redesigned 2026-08-19 for the four-row grid (Q1_melody, Q2_harmony,
Q3_timbre, Q4_rhythm) -- the old nine-row/tier grid's builders are gone.

Every build_* function is pure given (rng, cfg, ...) and returns a single
SymbolicItem describing BOTH poles of one comparison: `params` carries
whatever note/timing data is needed to render the base pole (gold_base)
and, deterministically, the content-edited pole (gold_content_edit). See
"The item" is the shared musical frame and the two poles are its two
concrete fillings.

All four ladders (100/200/400/700c melody, 100/200/300c harmony) are exact
semitone multiples, so nothing here ever needs sub-semitone pitch bend --
every pitch is a plain MIDI note number.
"""

import hashlib
import json
from dataclasses import dataclass, field

import music21


@dataclass(frozen=True)
class SymbolicItem:
    item_id: str
    task: str
    gold_base: str
    gold_content_edit: str
    content_key: str
    ladder_value: float | None
    ladder_unit: str | None
    params: dict = field(compare=False)


def to_music21_stream(notes, tempo_bpm=120):
    """Note list ({"midi", "start", "dur"} dicts, times in seconds) -> music21 Stream, for
    inspecting or exporting an item's symbolic content (e.g. `.show("midi")`, `.write("musicxml", ...)`).
    Rendering does not depend on it."""
    stream = music21.stream.Stream()
    stream.insert(0, music21.tempo.MetronomeMark(number=tempo_bpm))
    quarter = 60.0 / tempo_bpm
    for n in notes:
        note = music21.note.Note(int(n["midi"]))
        note.quarterLength = n["dur"] / quarter
        stream.insert(n["start"] / quarter, note)
    return stream


def _hash(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


content_hash = _hash  # public alias; make_render_sets.py hashes the content-edit pole with it


def note_and_bend(base_midi: int, cents_offset: float):
    """Decompose an arbitrary cents offset from `base_midi` into
    (final_midi_note, remainder_cents). Every ladder in this grid is an
    exact semitone multiple, so remainder is always 0 in practice; kept
    generic (and used by render.py) in case a future row needs it."""
    semitone_shift = round(cents_offset / 100.0)
    remainder = cents_offset - semitone_shift * 100.0
    return int(base_midi + semitone_shift), float(remainder)


# --- Q1_melody ---

def build_Q1_melody(rng, cfg, split, edit_cents: float):
    p = cfg.tasks["Q1_melody"]["params"]
    pool = p["reference_midi_pool"][split]
    n = p["n_notes"]
    edit_idx = p["edited_note_index"]
    edit_semitones = round(edit_cents / 100.0)

    motif_first = [int(rng.choice(pool)) for _ in range(n)]

    gold_base = rng.choice(["same", "different"])
    edit_sign = rng.choice([1, -1])
    motif_second_base = list(motif_first)
    if gold_base == "different":
        motif_second_base[edit_idx] = motif_first[edit_idx] + edit_sign * edit_semitones

    gold_edit = "different" if gold_base == "same" else "same"
    motif_second_edit = list(motif_first)
    if gold_edit == "different":
        motif_second_edit[edit_idx] = motif_first[edit_idx] + edit_sign * edit_semitones

    transposition_cents = float(rng.uniform(0.0, 200.0))
    params = {
        "note_dur_seconds": p["note_dur_seconds"],
        "motif_first": motif_first,
        "motif_second_base": motif_second_base,
        "motif_second_edit": motif_second_edit,
        "edited_note_index": edit_idx,
        "transposition_cents": transposition_cents,
        "is_catch": False,
    }
    content_key = _hash({"task": "Q1_melody", "m1": motif_first, "m2": motif_second_base})
    return SymbolicItem("Q1", "Q1_melody", gold_base, gold_edit, content_key,
                         edit_cents, "cents", params)


def build_Q1_catch(rng, cfg, split):
    """Identical-clip catch: base/invariance poles are
    bit-identical, gold 'same'. The equivariance pole still needs a real
    'different' answer (schema rule 3) -- uses the smallest ladder rung."""
    p = cfg.tasks["Q1_melody"]["params"]
    pool = p["reference_midi_pool"][split]
    n = p["n_notes"]
    edit_idx = p["edited_note_index"]
    smallest_cents = min(cfg.tasks["Q1_melody"]["ladder"]["values"])
    edit_semitones = round(smallest_cents / 100.0)
    edit_sign = rng.choice([1, -1])

    motif_first = [int(rng.choice(pool)) for _ in range(n)]
    motif_second_edit = list(motif_first)
    motif_second_edit[edit_idx] = motif_first[edit_idx] + edit_sign * edit_semitones

    params = {
        "note_dur_seconds": p["note_dur_seconds"],
        "motif_first": motif_first,
        "motif_second_base": list(motif_first),
        "motif_second_edit": motif_second_edit,
        "edited_note_index": edit_idx,
        "transposition_cents": float(rng.uniform(0.0, 200.0)),
        "is_catch": True,
    }
    content_key = _hash({"task": "Q1_melody_catch", "m1": motif_first})
    return SymbolicItem("Q1catch", "Q1_melody", "same", "different", content_key, None, None, params)


# --- Q2_harmony ---

_THIRD_CENTS = {"major": 400, "minor": 300}


def _third_shift_direction(quality_first):
    """major (400c) always shifts DOWN, minor (300c) always shifts UP --
    the only direction rule that keeps every rung (up to 300c) inside
    [100c, 600c], safely clear of the root (0c) and the fifth (700c) with
    margin to spare."""
    return -1 if quality_first == "major" else 1


def build_Q2_harmony(rng, cfg, split, third_shift_cents: float):
    p = cfg.tasks["Q2_harmony"]["params"]
    root = int(rng.choice(p["root_midi_pool"][split]))
    margin = p["third_margin_cents"]
    quality_first = rng.choice(["major", "minor"])
    third_first_cents = _THIRD_CENTS[quality_first]
    sign = _third_shift_direction(quality_first)

    gold_base = rng.choice(["same", "different"])
    third_second_base = third_first_cents if gold_base == "same" else third_first_cents + sign * third_shift_cents
    gold_edit = "different" if gold_base == "same" else "same"
    third_second_edit = third_first_cents if gold_edit == "same" else third_first_cents + sign * third_shift_cents

    for val in (third_second_base, third_second_edit):
        assert margin <= val <= 700 - margin, f"third at {val}c violates the {margin}c margin from root/fifth"

    transposition_cents = float(rng.uniform(0.0, 200.0))
    params = {
        "root_midi": root,
        "chord_dur_seconds": p["chord_dur_seconds"],
        "quality_first": quality_first,
        "third_first_cents": third_first_cents,
        "third_second_base_cents": third_second_base,
        "third_second_edit_cents": third_second_edit,
        "transposition_cents": transposition_cents,
        "is_catch": False,
    }
    content_key = _hash({"task": "Q2_harmony", "root": root, "q1": quality_first, "t2": third_second_base})
    return SymbolicItem("Q2", "Q2_harmony", gold_base, gold_edit, content_key,
                         third_shift_cents, "cents", params)


def build_Q2_catch(rng, cfg, split):
    p = cfg.tasks["Q2_harmony"]["params"]
    root = int(rng.choice(p["root_midi_pool"][split]))
    quality_first = rng.choice(["major", "minor"])
    third_first_cents = _THIRD_CENTS[quality_first]
    sign = _third_shift_direction(quality_first)
    smallest = min(cfg.tasks["Q2_harmony"]["ladder"]["values"])

    params = {
        "root_midi": root,
        "chord_dur_seconds": p["chord_dur_seconds"],
        "quality_first": quality_first,
        "third_first_cents": third_first_cents,
        "third_second_base_cents": third_first_cents,
        "third_second_edit_cents": third_first_cents + sign * smallest,
        "transposition_cents": float(rng.uniform(0.0, 200.0)),
        "is_catch": True,
    }
    content_key = _hash({"task": "Q2_harmony_catch", "root": root, "q1": quality_first})
    return SymbolicItem("Q2catch", "Q2_harmony", "same", "different", content_key, None, None, params)


# --- Q3_timbre ---

def build_Q3_timbre(rng, cfg, split, program_pair: tuple):
    p = cfg.tasks["Q3_timbre"]["params"]
    pool = p["reference_midi_pool"][split]
    n = p["n_notes"]
    programs = list(p["gm_programs"])
    melody = [int(rng.choice(pool)) for _ in range(n)]
    p1, p2 = program_pair
    gold_base = "same" if p1 == p2 else "different"

    if gold_base == "same":
        alt = rng.choice([x for x in programs if x != p1])
        program_pair_edit = (p1, alt)
        gold_edit = "different"
    else:
        program_pair_edit = (p1, p1)
        gold_edit = "same"

    params = {
        "note_dur_seconds": p["note_dur_seconds"],
        "melody": melody,
        "program_pair_base": (int(p1), int(p2)),
        "program_pair_edit": (int(program_pair_edit[0]), int(program_pair_edit[1])),
        "is_catch": False,
    }
    content_key = _hash({"task": "Q3_timbre", "melody": melody, "pair": params["program_pair_base"]})
    return SymbolicItem("Q3", "Q3_timbre", gold_base, gold_edit, content_key, None, None, params)


def build_Q3_catch(rng, cfg, split):
    p = cfg.tasks["Q3_timbre"]["params"]
    pool = p["reference_midi_pool"][split]
    n = p["n_notes"]
    programs = list(p["gm_programs"])
    melody = [int(rng.choice(pool)) for _ in range(n)]
    p1 = int(rng.choice(programs))
    alt = int(rng.choice([x for x in programs if x != p1]))

    params = {
        "note_dur_seconds": p["note_dur_seconds"],
        "melody": melody,
        "program_pair_base": (p1, p1),
        "program_pair_edit": (p1, alt),
        "is_catch": True,
    }
    content_key = _hash({"task": "Q3_timbre_catch", "melody": melody, "p1": p1})
    return SymbolicItem("Q3catch", "Q3_timbre", "same", "different", content_key, None, None, params)


# --- Q4_rhythm ---

def build_Q4_rhythm(rng, cfg, ratio: float):
    p = cfg.tasks["Q4_rhythm"]["params"]
    window = cfg.tasks["Q4_rhythm"]["window_seconds"]
    base_rate = p["base_rate_hz"]
    click_dur = p["click_dur_seconds"]

    gold_base = rng.choice(["first", "second"])
    gold_edit = "second" if gold_base == "first" else "first"

    period_ref = 1.0 / base_rate
    period_fast = 1.0 / (base_rate * ratio)
    phase_ref = round(rng.uniform(0.0, period_ref), 4)
    phase_fast = round(rng.uniform(0.0, period_fast), 4)

    def onsets(period, phase):
        out = []
        t = phase
        while t < window - click_dur:
            out.append(round(t, 4))
            t += period
        return out

    onsets_ref = onsets(period_ref, phase_ref)
    onsets_fast = onsets(period_fast, phase_fast)

    params = {
        "click_midi": p["click_midi"],
        "click_dur_seconds": click_dur,
        "window_seconds": window,
        "ratio": ratio,
        "base_rate_hz": base_rate,
        "onsets_ref": onsets_ref,
        "onsets_fast": onsets_fast,
        "phase_ref": phase_ref,
        "phase_fast": phase_fast,
        "is_catch": False,
    }
    content_key = _hash({"task": "Q4_rhythm", "ratio": ratio, "phase_ref": phase_ref,
                          "phase_fast": phase_fast, "gold_base": gold_base})
    return SymbolicItem("Q4", "Q4_rhythm", gold_base, gold_edit, content_key, ratio, "ratio", params)
