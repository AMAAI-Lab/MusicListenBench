(() => {
  const C = window.MLB_CONFIG;
  const ASPECTS = ["melody", "rhythm", "timbre", "harmony"];
  const NUMERIC = ASPECTS.concat(["overall", "acc_fs", "miss_rate", "false_flip_rate", "a_rate"]);
  const LOWER_IS_BETTER = ["miss_rate", "false_flip_rate"];

  /* ---------- Config-driven links and footer ---------- */
  document.querySelectorAll("[data-link]").forEach(a => {
    const url = C[a.dataset.link];
    if (url) a.href = url;
    else if (C.anonymous) a.hidden = true; // reviewers never see placeholders
    else { a.removeAttribute("href"); a.classList.add("todo"); a.title = "TODO: set " + a.dataset.link + " in assets/config.js"; }
  });
  const foot = document.getElementById("foot");
  if (foot) {
    const by = C.labUrl ? `<a href="${C.labUrl}">${C.lab}</a>` : C.lab;
    const src = C.csvSourceUrl || (C.githubRepo && C.githubRepo + "/blob/main/leaderboard.csv");
    const csv = src ? `Results live in <a href="${src}">leaderboard.csv</a>${C.anonymous ? "" : " on GitHub"}` : "";
    foot.innerHTML = `
    <div>${C.title} v${C.benchmarkVersion}. ${C.anonymous ? by : `Built by the ${by}`}.</div>
    <div>${csv}${C.contact ? `${csv ? " · " : ""}<a href="mailto:${C.contact}">${C.contact}</a>` : ""}${csv || C.contact ? "." : ""}</div>`;
  }

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
    const state = { rows: [], sort: "acc_fs", dir: -1, q: "", access: "all", verifiedOnly: false, chance: 50 };
    const tbody = board.querySelector("tbody");

    fetch(C.csvPath, { cache: "no-cache" })
      .then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
      .then(t => {
        state.rows = parseCSV(t).map(r => {
          NUMERIC.forEach(k => r[k] = parseFloat(r[k]));
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
        tbody.innerHTML = `<tr><td colspan="12" class="empty">Couldn't load ${esc(C.csvPath)}. If you opened this file directly from disk, run <code>python -m http.server</code> in the site folder instead.</td></tr>`;
      });

    const COLS = 12;
    const fmt = v => Number.isNaN(v) ? "—" : v.toFixed(1);
    const isTrained = r => !/^zero-shot$/i.test(r.method);

    function render() {
      const q = state.q.toLowerCase();
      const rows = state.rows.filter(r =>
        (r.type === "baseline" || state.access === "all" || r.access === state.access) &&
        (r.type === "baseline" || !state.verifiedOnly || r.verified === "yes") &&
        (!q || `${r.model} ${r.organization} ${r.method}`.toLowerCase().includes(q)));
      const key = state.sort;
      rows.sort((a, b) => {
        const av = a[key], bv = b[key];
        if (typeof av === "number") {
          if (Number.isNaN(av) || Number.isNaN(bv)) return Number.isNaN(av) - Number.isNaN(bv); // empty values last
          return (av - bv) * state.dir;
        }
        return String(av).localeCompare(String(bv)) * state.dir;
      });
      // Zero-shot models and models trained on the released training split are listed separately.
      const groups = [
        ["Zero-shot models", rows.filter(r => r.type === "model" && !isTrained(r))],
        ["Models trained on the released training split", rows.filter(r => r.type === "model" && isTrained(r))],
        ["References", rows.filter(r => r.type === "baseline")],
      ].filter(g => g[1].length);

      if (!groups.length) { tbody.innerHTML = `<tr><td colspan="${COLS}" class="empty">No results match these filters. Clear the search or show all models.</td></tr>`; return; }

      // Rank inside each group by FLIP/STAY accuracy.
      const rankOf = new Map();
      groups.filter(g => g[1][0].type === "model").forEach(([, g]) =>
        [...g].sort((a, b) => b.acc_fs - a.acc_fs).forEach((r, i) => rankOf.set(r, i + 1)));

      const rowHtml = r => {
        const isBase = r.type === "baseline";
        const bar = (k, cls = "num") => `<td class="${cls}" style="--c:${ASPECTS.includes(k) ? `var(--${k})` : "var(--ink)"};--chance:${state.chance}%">
            <div class="score"><span>${fmt(r[k])}</span><div class="bar"><i style="width:${r[k] || 0}%"></i></div></div></td>`;
        const plain = k => `<td class="num plain"><span>${fmt(r[k])}</span></td>`;
        const name = r.model_url ? `<a href="${esc(r.model_url)}">${esc(r.model)}</a>` : esc(r.model);
        const tags = isBase ? "" :
          `<span class="tag">${esc(r.access)}</span>` + (r.verified === "yes" ? "" : `<span class="tag self" title="Self-reported, not yet reproduced by maintainers">self-reported</span>`);
        const meta = isBase ? "Reference line" : [r.organization, r.params_b && `${r.params_b}B`].filter(Boolean).map(esc).join(", ");
        const src = r.source_url ? ` <a href="${esc(r.source_url)}">source</a>` : "";
        const note = isBase ? "" : (r.notes ? `<small class="note">${esc(r.notes)}</small>` : "");
        return `<tr class="${isBase ? "baseline" : ""}">
          <td class="rank">${isBase ? "" : rankOf.get(r)}</td>
          <td class="model"${isBase && r.notes ? ` title="${esc(r.notes)}"` : ""}><b>${name}</b><small>${meta}${src}</small>${note}${tags ? `<div class="tags">${tags}</div>` : ""}</td>
          <td class="method">${esc(r.method)}</td>
          ${ASPECTS.map(k => bar(k)).join("")}
          ${bar("overall", "num overall")}
          ${bar("acc_fs", "num overall")}
          ${plain("miss_rate")}${plain("false_flip_rate")}${plain("a_rate")}
        </tr>`;
      };

      tbody.innerHTML = groups.map(([title, g]) =>
        `<tr class="group"><th scope="colgroup" colspan="${COLS}">${esc(title)}</th></tr>` + g.map(rowHtml).join("")).join("");
    }

    board.querySelectorAll("th[data-sort] button").forEach(btn => btn.addEventListener("click", () => {
      const th = btn.closest("th"), k = th.dataset.sort;
      state.dir = state.sort === k ? -state.dir : (["model", "method"].concat(LOWER_IS_BETTER).includes(k) ? 1 : -1);
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
      rhythm: { mel: m => m.map(([t, p, d]) => [t * 0.8, p, d * 0.8]), ch: c => c.map(([t, ps, d]) => [t * 0.8, ps, d * 0.8]),
        changed: { mel: [0, 1, 2, 3, 4, 5, 6, 7], ch: [0, 1] },
        caption: "Rhythm items ask which clip is faster. Clip B plays the same notes at a faster tempo, so the answer is: clip B." },
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
      q.textContent = aspect === "rhythm" ? "Which clip is faster?" : `Same or different ${aspect}?`;
      pair.querySelectorAll(".aspects button").forEach(b => b.setAttribute("aria-pressed", b.dataset.aspect === aspect));
    }
    svg.setAttribute("viewBox", `0 0 ${W} ${top * 2 + laneH * 2 + gap}`);
    pair.querySelectorAll(".aspects button").forEach(b => b.addEventListener("click", () => draw(b.dataset.aspect)));
    draw("melody");
  }
})();
