"""Scans the MIDI of every Slakh2100 stem once and stores notes and metadata for item selection.

    SLAKH_DIR=/path/to/slakh2100_flac_redux python -m musiclistenbench.other_sets.scan_slakh
"""
import json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import yaml, pretty_midi
SLAKH = os.environ["SLAKH_DIR"]
OUT = os.path.join(os.environ.get("SLAKH_OUT", "generated/slakh"), "stems_index.pkl")
def scan(args):
    split, tr = args; d = f"{SLAKH}/{split}/{tr}"; rows = []
    try: meta = yaml.safe_load(open(f"{d}/metadata.yaml"))
    except Exception: return rows
    for s, m in (meta.get("stems") or {}).items():
        a, mid = f"{d}/stems/{s}.flac", f"{d}/MIDI/{s}.mid"
        if not (m.get("audio_rendered") and os.path.exists(a) and os.path.exists(mid)): continue
        try: pm = pretty_midi.PrettyMIDI(mid)
        except Exception: continue
        notes = sorted([(n.start, n.end, n.pitch, n.velocity) for i in pm.instruments for n in i.notes])
        rows.append(dict(split=split, track=tr, stem=s, audio=a, inst_class=m.get("inst_class"), program=m.get("program_num"),
                         program_name=m.get("midi_program_name"), plugin=m.get("plugin_name"), is_drum=bool(m.get("is_drum")), notes=notes))
    return rows
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    jobs = [(sp, t) for sp in ("train", "validation", "test") for t in sorted(os.listdir(f"{SLAKH}/{sp}"))]
    with ProcessPoolExecutor(64) as ex: rows = [r for rs in ex.map(scan, jobs, chunksize=4) for r in rs]
    pickle.dump(rows, open(OUT, "wb"))
    import collections
    print(len(rows), collections.Counter((r["split"], r["inst_class"]) for r in rows).most_common(60))
