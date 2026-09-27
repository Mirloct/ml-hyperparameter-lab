/* Gráficas adicionales: estabilidad, Mass-Volume y optimización de hiperparámetros. */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});
  const { clamp, fmt, esc } = ML;
  const C = ML.charts, H = C._h;
  const { lin, niceTicks, box, svgOpen, gridY, axisX, xTitle, yTitle, hover, dot, row } = H;

  const mix = (pct) => `color-mix(in srgb, var(--s1) ${Math.round(pct)}%, var(--surface))`;

  /* ---------- Matriz de Jaccard entre semillas ---------- */
  C.matrix = function (el, { values, labels, mean, k }) {
    const n = values.length, b = box(el, 0.9, 200, 280, { t: 30, r: 10, b: 16, l: 34 });
    const size = Math.min(b.iw, b.ih), cell = size / n, x0 = b.m.l + (b.iw - size) / 2, y0 = b.m.t;
    let cells = '';
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = values[i][j], dark = v > 0.6;
      cells += `<rect x="${x0 + j * cell + 1}" y="${y0 + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="4" style="fill:${mix(8 + v * 70)}"/>` +
        `<text x="${x0 + j * cell + cell / 2}" y="${y0 + i * cell + cell / 2 + 4}" text-anchor="middle" style="fill:${dark ? '#fff' : 'var(--ink)'};font-weight:600;font-size:11.5px">${i === j ? '1' : v.toFixed(2)}</text>`;
    }
    const lab = labels.map((l, i) => `<text x="${x0 + i * cell + cell / 2}" y="${y0 - 8}" text-anchor="middle">${l}</text><text x="${x0 - 8}" y="${y0 + i * cell + cell / 2 + 4}" text-anchor="end">${l}</text>`).join('');
    el.innerHTML = svgOpen(b, 'Matriz de Jaccard entre semillas') + lab + cells +
      `<text class="title" x="${x0 + size / 2}" y="${y0 + size + 13}" text-anchor="middle">Jaccard@${k} · media ${fmt(mean, 2)}</text>` +
      `<g class="cursor" style="display:none"><rect class="cur-band" x="0" y="0" width="${cell}" height="${cell}"/></g><rect class="hit" x="${x0}" y="${y0}" width="${size}" height="${size}"/></svg>`;
    const cur = el.querySelector('.cur-band');
    hover(el, (px, py) => {
      const j = Math.floor((px - x0) / cell), i = Math.floor((py - y0) / cell);
      if (i < 0 || j < 0 || i >= n || j >= n) return null;
      cur.setAttribute('x', x0 + j * cell); cur.setAttribute('y', y0 + i * cell);
      return `<span class="tip-h">${labels[i]} vs ${labels[j]}</span>${row('Jaccard@' + k, fmt(values[i][j], 3))}<span class="tip-sub">Fracción de las ${k} alertas que comparten las dos semillas.</span>`;
    });
  };

  /* ---------- Curva Mass-Volume ---------- */
  C.mvCurve = function (el, { alphas, mv, area }) {
    const b = box(el, 0.62, 190, 280, { t: 14, r: 12, b: 38, l: 44 });
    const x = lin(0.9, 1, b.m.l, b.m.l + b.iw), y = lin(0, 1, b.m.t + b.ih, b.m.t);
    const d = alphas.map((a, i) => `${i ? 'L' : 'M'}${x(a).toFixed(1)} ${y(mv[i]).toFixed(1)}`).join('');
    el.innerHTML = svgOpen(b, 'Curva Mass-Volume') + gridY(b, y, [0, 0.25, 0.5, 0.75, 1], (t) => t.toFixed(2)) +
      axisX(b, x, [0.9, 0.925, 0.95, 0.975, 1], (t) => t.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')) +
      `<line class="ref" x1="${x(0.9)}" y1="${y(0.9)}" x2="${x(1)}" y2="${y(1)}"/><text class="mut" x="${x(0.905)}" y="${y(0.9) + 18}">sin estructura (volumen ≈ masa)</text>` +
      `<path d="${d}" fill="none" stroke-width="2.2" stroke-linejoin="round" style="stroke:var(--s1)"/>` +
      xTitle(b, 'masa capturada α (fracción de los datos)') + yTitle(b, 'volumen del conjunto de nivel') +
      `<text class="ink" x="${b.m.l + b.iw - 6}" y="${b.m.t + b.ih - 10}" text-anchor="end">MV medio = ${fmt(area, 3)} (menor = mejor)</text>` +
      `<g class="cursor" style="display:none"><line class="cur-line" y1="${b.m.t}" y2="${b.m.t + b.ih}"/></g><rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cl = el.querySelector('.cur-line');
    hover(el, (px) => {
      const a = clamp(x.inv(px), 0.9, 0.999), i = clamp(Math.round((a - 0.9) / 0.001), 0, mv.length - 1);
      cl.setAttribute('x1', x(alphas[i])); cl.setAttribute('x2', x(alphas[i]));
      return `<span class="tip-h">Para capturar el ${(alphas[i] * 100).toFixed(1)} % de los datos…</span>${row('Volumen necesario', fmt(mv[i], 3))}<span class="tip-sub">Un buen score la logra con poco volumen: la masa está concentrada.</span>`;
    });
  };

  /* ---------- Paisaje ψ × (max_features, bootstrap) ---------- */
  C.landscape = function (el, { space, values, marks, oracle, dflt, lo, hi }) {
    // filas: combinaciones (mf, bs); columnas: ψ
    const rowsDef = [];
    for (const mf of space.mf) for (const bs of space.bs) rowsDef.push({ mf, bs });
    const nC = space.psi.length, nR = rowsDef.length;
    const b = box(el, 0.36, 170, 260, { t: 10, r: 10, b: 40, l: 92 });
    const cw = b.iw / nC, ch = b.ih / nR;
    let cells = '';
    rowsDef.forEach((r, ri) => space.psi.forEach((psi, ci) => {
      const v = values(psi, r.mf, r.bs), t = clamp((v - lo) / (hi - lo || 1), 0, 1);
      cells += `<rect x="${b.m.l + ci * cw + 1}" y="${b.m.t + ri * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="3" style="fill:${mix(6 + t * 82)}"/>`;
      if (cw > 30) cells += `<text x="${b.m.l + ci * cw + cw / 2}" y="${b.m.t + ri * ch + ch / 2 + 4}" text-anchor="middle" style="fill:${t > 0.55 ? '#fff' : 'var(--ink-2)'};font-size:10.5px">${v.toFixed(2).replace('0.', '.')}</text>`;
    }));
    const pos = (c) => ({ x: b.m.l + space.psi.indexOf(c.psi) * cw + cw / 2, y: b.m.t + rowsDef.findIndex((r) => r.mf === c.mf && r.bs === c.bs) * ch + ch / 2 });
    let overlay = '';
    if (dflt) { const p = pos(dflt); overlay += `<rect x="${p.x - cw / 2 + 2}" y="${p.y - ch / 2 + 2}" width="${cw - 4}" height="${ch - 4}" rx="3" fill="none" stroke-width="2" stroke-dasharray="3 2" style="stroke:var(--ink)"/>`; }
    if (oracle) { const p = pos(oracle); overlay += `<path d="M${p.x} ${p.y - 9}L${p.x + 9} ${p.y}L${p.x} ${p.y + 9}L${p.x - 9} ${p.y}Z" fill="none" stroke-width="2" style="stroke:var(--s2)"/>`; }
    (marks || []).forEach((m, i) => {
      const p = pos(m.cfg);
      overlay += `<circle cx="${p.x}" cy="${p.y}" r="${m.final ? 9 : 7}" style="fill:var(--surface);stroke:var(--ink);stroke-width:${m.final ? 2.6 : 1.3}"/><text x="${p.x}" y="${p.y + 3.5}" text-anchor="middle" style="fill:var(--ink);font-size:9.5px;font-weight:600">${i + 1}</text>`;
    });
    const rl = rowsDef.map((r, ri) => `<text x="${b.m.l - 8}" y="${b.m.t + ri * ch + ch / 2 + 4}" text-anchor="end">mf=${r.mf} · bs=${r.bs ? 'T' : 'F'}</text>`).join('');
    const cl = space.psi.map((p, ci) => ((ci % 2 === 0 || cw > 34) ? `<text x="${b.m.l + ci * cw + cw / 2}" y="${b.m.t + b.ih + 15}" text-anchor="middle">${p}</text>` : '')).join('');
    el.innerHTML = svgOpen(b, 'Paisaje de hiperparámetros: AP real según max_samples, max_features y bootstrap') + rl + cl + cells + overlay +
      xTitle(b, 'max_samples ψ  →') +
      `<g class="cursor" style="display:none"><rect class="cur-band" x="0" y="0" width="${cw}" height="${ch}"/></g><rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cb = el.querySelector('.cur-band');
    hover(el, (px, py) => {
      const ci = Math.floor((px - b.m.l) / cw), ri = Math.floor((py - b.m.t) / ch);
      if (ci < 0 || ri < 0 || ci >= nC || ri >= nR) return null;
      cb.setAttribute('x', b.m.l + ci * cw); cb.setAttribute('y', b.m.t + ri * ch);
      const psi = space.psi[ci], r = rowsDef[ri], evs = (marks || []).map((m, i) => ({ m, i })).filter((o) => o.m.cfg.psi === psi && o.m.cfg.mf === r.mf && o.m.cfg.bs === r.bs);
      return `<span class="tip-h">ψ = ${psi} · max_features = ${r.mf} · bootstrap = ${r.bs ? 'True' : 'False'}</span>${row('AP real (T = 300)', fmt(values(psi, r.mf, r.bs), 3))}` +
        (evs.length ? `<span class="tip-sub">Evaluada en la búsqueda: paso ${evs.map((o) => o.i + 1).join(', ')}</span>` : '');
    });
  };

  /* ---------- Curvas de búsqueda con bandas (mediana y rango intercuartil) ---------- */
  C.searchCurves = function (el, { grid, series, oracle, dflt, budget }) {
    const b = box(el, 0.5, 220, 320, { t: 14, r: 92, b: 40, l: 50 });
    let mn = Infinity, mx = -Infinity;
    series.forEach((s) => { [s.med, s.lo, s.hi].forEach((a) => a.forEach((v) => { if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } })); });
    mn = Math.min(mn, dflt); mx = Math.max(mx, oracle);
    const pad = Math.max(0.01, (mx - mn) * 0.08);
    const y0 = clamp(Math.floor((mn - pad) * 50) / 50, 0, 0.98), y1 = clamp(Math.ceil((mx + pad) * 50) / 50, y0 + 0.05, 1);
    const x = lin(grid[0], budget, b.m.l, b.m.l + b.iw), y = lin(y0, y1, b.m.t + b.ih, b.m.t);
    let body = '', labels = [];
    series.forEach((s) => {
      const pts = grid.map((g, i) => [g, s.med[i], s.lo[i], s.hi[i]]).filter((p) => Number.isFinite(p[1]));
      if (!pts.length) return;
      if (s.band) {
        const up = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)} ${y(p[3]).toFixed(1)}`).join('');
        const dn = pts.slice().reverse().map((p) => `L${x(p[0]).toFixed(1)} ${y(p[2]).toFixed(1)}`).join('');
        body += `<path d="${up}${dn}Z" style="fill:${s.color};opacity:.14"/>`;
      }
      body += `<path d="${pts.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)} ${y(p[1]).toFixed(1)}`).join('')}" fill="none" stroke-width="2.2" stroke-linejoin="round" style="stroke:${s.color}"/>`;
      const last = pts[pts.length - 1];
      body += `<circle cx="${x(last[0])}" cy="${y(last[1])}" r="3.6" style="fill:${s.color};stroke:var(--surface);stroke-width:1.5"/>`;
      labels.push({ name: s.name, y: y(last[1]) });
    });
    labels.sort((a, c) => a.y - c.y);
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
    const lab = labels.map((l) => `<text class="ink" x="${b.m.l + b.iw + 8}" y="${l.y + 4}">${esc(l.name)}</text>`).join('');
    const ref = (v, txt) => (v >= y0 && v <= y1 ? `<line class="ref" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${y(v)}" y2="${y(v)}"/><text class="mut" x="${b.m.l + 6}" y="${y(v) - 5}">${txt}</text>` : '');
    const xt = niceTicks(grid[0], budget, 5);
    el.innerHTML = svgOpen(b, 'Calidad real de la configuración elegida según el costo de la búsqueda') +
      gridY(b, y, niceTicks(y0, y1, 4), (t) => t.toFixed(2)) + axisX(b, x, xt, (t) => Math.round(t)) +
      ref(oracle, `mejor posible ${fmt(oracle, 3)}`) + ref(dflt, `valores por defecto ${fmt(dflt, 3)}`) + body + lab +
      xTitle(b, 'costo acumulado (árboles entrenados)') + yTitle(b, 'AP real de lo elegido') +
      `<g class="cursor" style="display:none"><line class="cur-line" y1="${b.m.t}" y2="${b.m.t + b.ih}"/></g><rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cl = el.querySelector('.cur-line');
    hover(el, (px) => {
      const g = x.inv(px); let gi = 0, bd = Infinity;
      grid.forEach((v, i) => { const d = Math.abs(v - g); if (d < bd) { bd = d; gi = i; } });
      cl.setAttribute('x1', x(grid[gi])); cl.setAttribute('x2', x(grid[gi]));
      let h = `<span class="tip-h">Tras ${Math.round(grid[gi])} árboles entrenados</span>`;
      series.forEach((s) => { if (Number.isFinite(s.med[gi])) h += row(dot(s.color) + esc(s.name), `${fmt(s.med[gi], 3)} <small style="opacity:.7">[${fmt(s.lo[gi], 2)}–${fmt(s.hi[gi], 2)}]</small>`); });
      return h + '<span class="tip-sub">Mediana [rango intercuartil] de 60 búsquedas simuladas</span>';
    });
  };

  /* ---------- Proxy sin etiquetas vs AP real ---------- */
  C.proxyScatter = function (el, { title, pts, rho, xLabel, dflt, bestAP, oracleAP }) {
    const b = box(el, 0.92, 190, 260, { t: 44, r: 12, b: 40, l: 50 });
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const xl = Math.min(...xs), xh = Math.max(...xs), yl = Math.min(...ys), yh = Math.max(...ys);
    const px = lin(xl - (xh - xl) * 0.06, xh + (xh - xl) * 0.06 || 1, b.m.l, b.m.l + b.iw), py = lin(Math.max(0, yl - 0.03), Math.min(1, yh + 0.03), b.m.t + b.ih, b.m.t);
    const dots = pts.map((p) => `<circle cx="${px(p.x).toFixed(1)}" cy="${py(p.y).toFixed(1)}" r="3.4" style="fill:${p.mf === 1 ? 'var(--s1)' : 'var(--s2)'};opacity:.85;stroke:var(--surface);stroke-width:1"/>`).join('');
    const dp = dflt ? `<circle cx="${px(dflt.x)}" cy="${py(dflt.y)}" r="6.5" fill="none" stroke-width="1.8" style="stroke:var(--ink)"/>` : '';
    const regret = oracleAP - bestAP, verdict = regret <= 0.02 && rho >= 0.5 ? 'guía bien' : regret > 0.05 ? 'falla en el óptimo' : 'guía a medias';
    el.innerHTML = svgOpen(b, `${title} frente al AP real`) + gridY(b, py, niceTicks(py.inv(b.m.t + b.ih), py.inv(b.m.t), 3), (t) => t.toFixed(2)) +
      `<line class="axis" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${b.m.t + b.ih}" y2="${b.m.t + b.ih}"/>` + dots + dp +
      `<text class="ink" x="${b.m.l}" y="${b.m.t - 28}">${esc(title)}</text><text class="mut" x="${b.m.l}" y="${b.m.t - 15}">ρ = ${fmt(rho, 2)} · su máximo rinde AP ${fmt(bestAP, 2)} (óptimo ${fmt(oracleAP, 2)})</text><text class="ink" x="${b.m.l}" y="${b.m.t - 2}" style="fill:${verdict === "guía bien" ? "var(--good)" : verdict === "falla en el óptimo" ? "var(--bad)" : "var(--ink-2)"}">${verdict === "guía bien" ? "✓" : verdict === "falla en el óptimo" ? "✕" : "~"} ${verdict}</text>` +
      xTitle(b, esc(xLabel)) + yTitle(b, 'AP real') +
      `<g class="cursor" style="display:none"><circle class="cur-dot" r="6" style="fill:none;stroke:var(--ink);stroke-width:2"/></g><rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cd = el.querySelector('.cur-dot');
    hover(el, (mx, my) => {
      let best = null, bd = 400;
      pts.forEach((p) => { const d = (px(p.x) - mx) ** 2 + (py(p.y) - my) ** 2; if (d < bd) { bd = d; best = p; } });
      if (!best) return null;
      cd.setAttribute('cx', px(best.x)); cd.setAttribute('cy', py(best.y));
      return `<span class="tip-h">ψ = ${best.psi} · max_features = ${best.mf} · bootstrap = ${best.bs ? 'True' : 'False'}</span>${row(esc(xLabel), fmt(best.raw, 3))}${row('AP real', fmt(best.y, 3))}`;
    });
  };
})();

/* Redibujo automático: cada gráfica se vuelve a dibujar cuando cambia el ancho de su contenedor. */
(function () {
  'use strict';
  const C = window.MLLab.charts;
  if (C._observed || !('ResizeObserver' in window)) return;
  C._observed = true;
  const ro = new ResizeObserver((entries) => {
    entries.forEach((e) => {
      const w = Math.round(e.contentRect.width), el = e.target;
      if (w > 0 && el._lastW !== w) { el._lastW = w; if (el._redraw) el._redraw(); }
    });
  });
  Object.keys(C).filter((k) => typeof C[k] === 'function' && k[0] !== '_').forEach((k) => {
    const f = C[k];
    C[k] = function (el, args) {
      el._redraw = () => f(el, args);
      el._lastW = Math.round(el.clientWidth);
      f(el, args);
      ro.observe(el);
    };
  });
})();
