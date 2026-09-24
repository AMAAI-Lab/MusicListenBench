(() => {
  const C = window.MLB_CONFIG;
  const ASPECTS = ["melody", "rhythm", "timbre", "harmony"];

  /* ---------- Config-driven links and footer ---------- */
  document.querySelectorAll("[data-link]").forEach(a => {
    const url = C[a.dataset.link];
    if (url) a.href = url;
    else { a.removeAttribute("href"); a.classList.add("todo"); a.title = "TODO: set " + a.dataset.link + " in assets/config.js"; }
  });
  const foot = document.getElementById("foot");
  if (foot) foot.innerHTML = `
    <div>${C.title} v${C.benchmarkVersion}. Built by the ${C.labUrl ? `<a href="${C.labUrl}">${C.lab}</a>` : C.lab}.</div>
    <div>Results live in <a href="${C.githubRepo}/blob/main/leaderboard.csv">leaderboard.csv</a> on GitHub${C.contact ? ` · <a href="mailto:${C.contact}">${C.contact}</a>` : ""}.</div>`;

  /* ---------- CSV parsing (handles quoted fields) ---------- */
  function parseCSV(text) {
    const rows = []; let row = [], f = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) {
        if (c === '"' && text[i + 1] === '"') { f += '"'; i++; }
        else if (c === '"') q = false;
        else f += c;
      } else if (c === '"') q = true;
      else if (c === ",") { row.push(f); f = ""; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(f); f = ""; if (row.some(x => x !== "")) rows.push(row); row = [];
      } else f += c;
    }
    row.push(f); if (row.some(x => x !== "")) rows.push(row);
    const head = rows.shift();
    return rows.map(r => Object.fromEntries(head.map((h, i) => [h, (r[i] ?? "").trim()])));
  }
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ---------- Leaderboard ---------- */
  const board = document.getElementById("board");
  if (board) {
    const state = { rows: [], sort: "overall", dir: -1, q: "", access: "all", verifiedOnly: false, chance: 50 };
    const tbody = board.querySelector("tbody");

    fetch(C.csvPath, { cache: "no-cache" })
      .then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then(t => {
        state.rows = parseCSV(t).map(r => {
          ASPECTS.concat("overall").forEach(k => r[k] = parseFloat(r[k]));
          return r;
        });
        const base = state.rows.find(r => r.type === "baseline" && /chance/i.test(r.model));
        if (base) state.chance = base.overall;
        const models = state.rows.filter(r => r.type === "model");
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set("stat-models", new Set(models.map(r => r.model)).size);
        set("stat-updated", models.map(r => r.date).sort().pop() || "—");
        render();
      })
      .catch(() => {
        tbody.innerHTML = `<tr><td colspan="8" class="empty">Couldn't load ${esc(C.csvPath)}. If you opened this file directly from disk, run <code>python -m http.server</code> in the site folder instead.</td></tr>`;
      });

    function render() {
      const q = state.q.toLowerCase();
      let rows = state.rows.filter(r =>
        (r.type === "baseline" || state.access === "all" || r.access === state.access) &&
        (r.type === "baseline" || !state.verifiedOnly || r.verified === "yes") &&
        (!q || `${r.model} ${r.organization} ${r.method}`.toLowerCase().includes(q)));
      const key = state.sort;
      rows.sort((a, b) => {
        const av = a[key], bv = b[key];
        return typeof av === "number" ? (av - bv) * state.dir : String(av).localeCompare(String(bv)) * state.dir;
      });
      let rank = 0;
      const byOverall = [...rows].filter(r => r.type === "model").sort((a, b) => b.overall - a.overall);
      const rankOf = new Map(byOverall.map(r => [r, ++rank]));

      if (!rows.length) { tbody.innerHTML = `<tr><td colspan="8" class="empty">No results match these filters. Clear the search or show all models.</td></tr>`; return; }

      tbody.innerHTML = rows.map(r => {
        const isBase = r.type === "baseline";
        const cell = (k, cls = "num") => `<td class="${cls}" style="--c:var(--${k === "overall" ? "ink" : k});--chance:${state.chance}%">
            <div class="score"><span>${r[k].toFixed(1)}</span><div class="bar"><i style="width:${r[k]}%"></i></div></div></td>`;
        const name = r.model_url ? `<a href="${esc(r.model_url)}">${esc(r.model)}</a>` : esc(r.model);
        const tags = isBase ? "" :
          `<span class="tag">${esc(r.access)}</span>` + (r.verified === "yes" ? "" : `<span class="tag self" title="Self-reported, not yet reproduced by maintainers">self-reported</span>`);
        const meta = isBase ? "Reference line" : [r.organization, r.params_b && `${r.params_b}B`].filter(Boolean).map(esc).join(", ");
        const src = r.source_url ? ` <a href="${esc(r.source_url)}">source</a>` : "";
        return `<tr class="${isBase ? "baseline" : ""}">
          <td class="rank">${isBase ? "" : rankOf.get(r)}</td>
          <td class="model"><b>${name}</b><small>${meta}${src}</small>${tags ? `<div class="tags">${tags}</div>` : ""}</td>
          <td class="method">${esc(r.method)}</td>
          ${ASPECTS.map(k => cell(k)).join("")}
          ${cell("overall", "num overall")}
        </tr>`;
      }).join("");
    }

    board.querySelectorAll("th[data-sort] button").forEach(btn => btn.addEventListener("click", () => {
      const th = btn.closest("th"), k = th.dataset.sort;
      state.dir = state.sort === k ? -state.dir : (["model", "method"].includes(k) ? 1 : -1);
      state.sort = k;
      board.querySelectorAll("th[data-sort]").forEach(h => h.removeAttribute("aria-sort"));
      th.setAttribute("aria-sort", state.dir === -1 ? "descending" : "ascending");
      render();
    }));
    document.getElementById("search")?.addEventListener("input", e => { state.q = e.target.value; render(); });
    document.querySelectorAll("#access button").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll("#access button").forEach(x => x.setAttribute("aria-pressed", x === b));
      state.access = b.dataset.access; render();
    }));
    document.getElementById("verified")?.addEventListener("change", e => { state.verifiedOnly = e.target.checked; render(); });
  }

  /* ---------- Pair widget: two clips, one attribute changed ---------- */
  const pair = document.getElementById("pair");
  if (pair) {
    const W = 520, laneH = 150, gap = 18, top = 4, steps = 16, rows = 13;
    const x = t => 44 + t * ((W - 56) / steps);
    const y = (lane, p) => top + lane * (laneH + gap) + (rows - 1 - p) * (laneH / rows);
    const cw = (W - 56) / steps, rh = laneH / rows;

    // Clip A. Melody notes: [onset, pitch row, duration]. Chords: [onset, rows, duration].
    const melody = [[0, 7, 2], [2, 9, 2], [4, 11, 2], [6, 9, 1], [7, 7, 1], [8, 6, 2], [10, 7, 2], [12, 9, 4]];
    const chords = [[0, [0, 2], 8], [8, [1, 3], 8]];

    const variants = {
      melody: { mel: m => m.map((n, i) => i === 4 ? [7, 12, 1] : n), changed: { mel: [4] },
        caption: "In clip B one note moves up. Rhythm, timbre and harmony are identical, so the answer is: different melody." },
      rhythm: { mel: m => m.map((n, i) => i === 2 ? [4, 11, 1] : i === 3 ? [5, 9, 2] : n), changed: { mel: [2, 3] },
        caption: "Clip B plays the same pitches with different timing. The answer is: different rhythm." },
      timbre: { timbre: true, changed: { mel: [0, 1, 2, 3, 4, 5, 6, 7], ch: [0, 1] },
        caption: "Every note is the same, played on a different instrument. The answer is: different timbre." },
      harmony: { ch: c => c.map((n, i) => i === 1 ? [8, [0, 4], 8] : n), changed: { ch: [1] },
        caption: "The melody is unchanged, but the second chord underneath it changes. The answer is: different harmony." },
    };

    const svg = pair.querySelector("svg");
    const caption = pair.querySelector(".pair-caption");
    const q = pair.querySelector(".pair-q");

    function draw(aspect) {
      const v = variants[aspect];
      const clips = [
        { mel: melody, ch: chords, changed: { mel: [], ch: [] }, timbre: false },
        { mel: v.mel ? v.mel(melody) : melody, ch: v.ch ? v.ch(chords) : chords,
          changed: { mel: v.changed.mel || [], ch: v.changed.ch || [] }, timbre: !!v.timbre },
      ];
      let out = `<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="5" height="5" fill="var(--${aspect})" opacity=".35"/><line x1="0" y1="0" x2="0" y2="5" stroke="var(--${aspect})" stroke-width="2.5"/></pattern></defs>`;
      clips.forEach((clip, lane) => {
        const ly = top + lane * (laneH + gap);
        out += `<text class="lane-label" x="0" y="${ly + laneH / 2 + 4}">${lane ? "Clip B" : "Clip A"}</text>`;
        out += `<rect x="44" y="${ly}" width="${W - 56}" height="${laneH}" rx="8" fill="var(--paper)"/>`;
        for (let b = 4; b < steps; b += 4) out += `<line x1="${x(b)}" x2="${x(b)}" y1="${ly + 4}" y2="${ly + laneH - 4}" stroke="var(--rule)"/>`;
        clip.ch.forEach(([t, ps, d], i) => ps.forEach(p => {
          const hit = clip.changed.ch.includes(i);
          const fill = hit ? (clip.timbre ? "url(#hatch)" : `var(--${aspect})`) : "var(--muted)";
          out += `<rect class="note" x="${x(t) + 2}" y="${y(lane, p) + .5}" width="${d * cw - 4}" height="${rh - 1}" rx="3" fill="${fill}" opacity="${hit ? .9 : .28}"/>`;
        }));
        clip.mel.forEach(([t, p, d], i) => {
          const hit = clip.changed.mel.includes(i);
          const fill = hit ? (clip.timbre ? "url(#hatch)" : `var(--${aspect})`) : "var(--ink)";
          out += `<rect class="note" x="${x(t) + 2}" y="${y(lane, p) + .5}" width="${d * cw - 4}" height="${rh - 1}" rx="3" fill="${fill}"/>`;
        });
      });
      svg.innerHTML = out;
      caption.textContent = variants[aspect].caption;
      q.textContent = `Same or different ${aspect}?`;
      pair.querySelectorAll(".aspects button").forEach(b => b.setAttribute("aria-pressed", b.dataset.aspect === aspect));
    }
    svg.setAttribute("viewBox", `0 0 ${W} ${top * 2 + laneH * 2 + gap}`);
    pair.querySelectorAll(".aspects button").forEach(b => b.addEventListener("click", () => draw(b.dataset.aspect)));
    draw("melody");
  }
})();
