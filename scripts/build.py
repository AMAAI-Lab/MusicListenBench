#!/usr/bin/env python3
"""Build both leaderboard sites from site/ and leaderboard.csv.

  _site/  public version (GitHub Pages + HF Space)
  _anon/  anonymous reviewer version (Cloudflare Pages)

The anonymous build fails if any identifying term survives, so a leak
stops the deploy instead of reaching reviewers.

Usage: python scripts/build.py
"""
import csv, io, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE, CSV = ROOT / "site", ROOT / "leaderboard.csv"

# Anything that could identify the authors. Case-insensitive substring match.
# TODO: add every author's surname, and any other giveaway (grant names, emails).
LEAK_TERMS = [
    "AMAAI", "amaai-lab", "SUTD", "Singapore University",
    "sleeping-ai",  # style reference, not needed in either build
]

ANON_BRAND = "Anonymous submission"
ANON_SUBMIT = """<section class="block" style="border-top:0">
    <h2>Submissions open after review</h2>
    <p class="sub">This is the anonymous version of the leaderboard for peer review.
    The public leaderboard, with instructions for adding results, will be linked
    from the paper once the review period ends.</p>
  </section>"""
ANON_CITE = """<section class="block" id="cite">
    <h2>Cite</h2>
    <p class="sub">Citation details will be added after the review period.</p>
  </section>"""


def fresh(dest: Path):
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SITE, dest)
    (dest / "assets" / "config.anon.js").unlink(missing_ok=True)


def anonymize_csv(text: str) -> str:
    rows = list(csv.DictReader(io.StringIO(text)))
    fields = list(rows[0].keys()) if rows else []
    hit = lambda v: any(t.lower() in v.lower() for t in LEAK_TERMS)
    for r in rows:
        if hit(r["organization"]):
            r["organization"] = "Anonymous (this paper)"
        for k in ("model_url", "source_url"):
            if hit(r[k]):
                r[k] = ""
        if r["verified"] == "yes" or hit(r["submitted_by"]):
            r["submitted_by"] = "maintainers"
        if hit(r["notes"]):
            r["notes"] = ""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    return out.getvalue()


def build_public():
    dest = ROOT / "_site"
    fresh(dest)
    shutil.copy(CSV, dest / "leaderboard.csv")
    print(f"public build -> {dest}")


def build_anon():
    dest = ROOT / "_anon"
    fresh(dest)
    shutil.copy(SITE / "assets" / "config.anon.js", dest / "assets" / "config.js")
    (dest / "leaderboard.csv").write_text(anonymize_csv(CSV.read_text(encoding="utf-8")), encoding="utf-8")

    for page in dest.glob("*.html"):
        h = page.read_text(encoding="utf-8")
        h = re.sub(r'(<a class="brand"[^>]*>\s*<b>[^<]*</b>\s*<span>)[^<]*(</span>)', rf"\g<1>{ANON_BRAND}\g<2>", h)
        h = re.sub(r'<section class="block" id="cite">.*?</section>', ANON_CITE, h, flags=re.S)
        h = h.replace('<a class="btn" href="submit.html">Submit a result</a>', "")
        h = re.sub(r'\s*<a href="submit.html"[^>]*>Submit</a>', "", h)
        h = h.replace("<head>", '<head>\n<meta name="robots" content="noindex, nofollow">', 1)
        if page.name == "submit.html":
            h = re.sub(r'<main id="main" class="wrap">.*?</main>',
                       f'<main id="main" class="wrap">\n  {ANON_SUBMIT}\n</main>', h, flags=re.S)
        page.write_text(h, encoding="utf-8")
    (dest / "robots.txt").write_text("User-agent: *\nDisallow: /\n")

    leaks, todos = [], []
    for f in dest.rglob("*"):
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for t in LEAK_TERMS:
            for m in re.finditer(re.escape(t), text, re.I):
                line = text.count("\n", 0, m.start()) + 1
                leaks.append(f"{f.relative_to(dest)}:{line}: '{t}'")
        if f.suffix in (".html", ".csv") and "TODO" in text:
            todos.append(str(f.relative_to(dest)))
    for t in sorted(set(todos)):
        print(f"::warning::anonymous build still contains TODO placeholders in {t}")
    if leaks:
        print("::error::anonymous build contains identifying terms:")
        print("\n".join("  " + l for l in leaks))
        sys.exit(1)
    print(f"anonymous build -> {dest} (leak check passed)")


if __name__ == "__main__":
    build_public()
    build_anon()
