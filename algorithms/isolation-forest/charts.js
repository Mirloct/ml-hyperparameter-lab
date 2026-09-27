/* Gráficas SVG a mano (sin librerías). Los colores viven en CSS (var(--s1) …) para que el tema cambie sin redibujar. */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});
  const { clamp, fmt, esc } = ML;
  const C = (ML.charts = {});

  /* ---------- helpers ---------- */
  const lin = (d0, d1, r0, r1) => {
    const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
    f.inv = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * (d1 - d0);
    return f;
  };
  const logs = (d0, d1, r0, r1) => {
    const l0 = Math.log(d0), l1 = Math.log(d1);
    const f = (v) => r0 + ((Math.log(v) - l0) / (l1 - l0 || 1)) * (r1 - r0);
    f.inv = (p) => Math.exp(l0 + ((p - r0) / (r1 - r0 || 1)) * (l1 - l0));
    return f;
  };
  function niceTicks(lo, hi, n = 5) {
    const raw = (hi - lo) / n || 1;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function box(el, aspect, minH, maxH, m) {
    const w = Math.max(230, el.clientWidth || 320);
    const h = clamp(Math.round(w * aspect), minH, maxH);
    return { w, h, m, iw: w - m.l - m.r, ih: h - m.t - m.b };
  }
  const svgOpen = (b, label) =>
    `<svg class="ch" width="${b.w}" height="${b.h}" viewBox="0 0 ${b.w} ${b.h}" role="img" aria-label="${esc(label)}">`;
  const gridY = (b, y, ticks, f) =>
    ticks.map((t) => `<line class="grid" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${y(t)}" y2="${y(t)}"/><text x="${b.m.l - 7}" y="${y(t) + 4}" text-anchor="end">${f ? f(t) : t}</text>`).join('');
  const axisX = (b, x, ticks, f) =>
    `<line class="axis" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${b.m.t + b.ih}" y2="${b.m.t + b.ih}"/>` +
    ticks.map((t) => `<line class="axis" x1="${x(t)}" x2="${x(t)}" y1="${b.m.t + b.ih}" y2="${b.m.t + b.ih + 4}"/><text x="${x(t)}" y="${b.m.t + b.ih + 17}" text-anchor="middle">${f ? f(t) : t}</text>`).join('');
  const xTitle = (b, s) => `<text class="title" x="${b.m.l + b.iw / 2}" y="${b.h - 4}" text-anchor="middle">${s}</text>`;
  const yTitle = (b, s) => `<text class="title" transform="translate(11 ${b.m.t + b.ih / 2}) rotate(-90)" text-anchor="middle">${s}</text>`;

  /** Conecta el hover de una gráfica: fn(px, py) → html del tooltip (o null) y puede mover el cursor. */
  function hover(el, fn) {
    const hit = el.querySelector('.hit');
    if (!hit) return;
    const svg = el.querySelector('svg');
    const cur = el.querySelector('.cursor');
    const move = (e) => {
      const r = svg.getBoundingClientRect();
      const html = fn(e.clientX - r.left, e.clientY - r.top);
      if (cur) cur.style.display = html ? '' : 'none';
      if (html) ML.tip.show(html, e.clientX, e.clientY); else ML.tip.hide();
    };
    hit.addEventListener('pointermove', move);
    hit.addEventListener('pointerdown', move);
    hit.addEventListener('pointerleave', () => { ML.tip.hide(); if (cur) cur.style.display = 'none'; });
  }
  const dot = (v) => `<span class="dot" style="background:${v}"></span>`;
  const row = (k, v) => `<span class="tip-row"><span>${k}</span><b>${v}</b></span>`;

  /* ---------- 1. Distribución de scores por clase ---------- */
  C.scoreHist = function (el, { scores, y, thr, hasBoth }) {
    const b = box(el, 0.56, 190, 300, { t: 22, r: 12, b: 36, l: 44 });
    let mn = Infinity, mx = -Infinity;
    for (const s of scores) { if (s < mn) mn = s; if (s > mx) mx = s; }
    const lo = Math.floor((mn - 0.005) * 20) / 20, hi = Math.ceil((mx + 0.005) * 20) / 20;
    const nb = 26, bw = (hi - lo) / nb;
    const c0 = new Array(nb).fill(0), c1 = new Array(nb).fill(0);
    let t0 = 0, t1 = 0;
    scores.forEach((s, i) => {
      const k = clamp(Math.floor((s - lo) / bw), 0, nb - 1);
      if (y[i]) { c1[k]++; t1++; } else { c0[k]++; t0++; }
    });
    const share = (c, t) => (t ? c / t : 0);
    let ymax = 0;
    for (let k = 0; k < nb; k++) ymax = Math.max(ymax, share(c0[k], t0), share(c1[k], t1));
    ymax = Math.min(1, Math.ceil((ymax + 0.005) * 20) / 20);
    const x = lin(lo, hi, b.m.l, b.m.l + b.iw), yy = lin(0, ymax, b.m.t + b.ih, b.m.t);
    const slot = b.iw / nb, barW = Math.max(2, slot * (hasBoth ? 0.44 : 0.8));
    let bars = '';
    for (let k = 0; k < nb; k++) {
      const x0 = b.m.l + k * slot + slot / 2;
      const top0 = yy(share(c0[k], t0)), top1 = yy(share(c1[k], t1));
      if (hasBoth) {
        bars += `<rect x="${x0 - barW - 0.5}" y="${top0}" width="${barW}" height="${b.m.t + b.ih - top0}" rx="1.5" style="fill:var(--s1)"/>`;
        bars += `<rect x="${x0 + 0.5}" y="${top1}" width="${barW}" height="${b.m.t + b.ih - top1}" rx="1.5" style="fill:var(--s2)"/>`;
      } else {
        bars += `<rect x="${x0 - barW / 2}" y="${top0}" width="${barW}" height="${b.m.t + b.ih - top0}" rx="1.5" style="fill:var(--s1)"/>`;
      }
    }
    const yt = niceTicks(0, ymax, 4), xt = niceTicks(lo, hi, 6);
    const thrIn = thr >= lo && thr <= hi;
    const thrTxtRight = x(thr) < b.m.l + b.iw - 90;
    const ref = thr !== 0.5 && lo <= 0.5 && hi >= 0.5
      ? `<line class="ref" x1="${x(0.5)}" x2="${x(0.5)}" y1="${b.m.t}" y2="${b.m.t + b.ih}"/><text x="${x(0.5) + 4}" y="${b.m.t + b.ih - 6}" class="mut">0.5</text>` : '';
    const line = thrIn
      ? `<line class="thr" x1="${x(thr)}" x2="${x(thr)}" y1="${b.m.t - 6}" y2="${b.m.t + b.ih}"/><text class="ink" x="${x(thr) + (thrTxtRight ? 5 : -5)}" y="${b.m.t - 9}" text-anchor="${thrTxtRight ? 'start' : 'end'}">umbral ${fmt(thr, 3)}</text>` : '';
    el.innerHTML = svgOpen(b, 'Distribución de scores de anomalía por clase real') +
      gridY(b, yy, yt, (t) => Math.round(t * 100) + '%') + axisX(b, x, xt, (t) => t.toFixed(2)) + bars + ref + line +
      xTitle(b, 'score de anomalía  s(x)  →  más anómalo') + yTitle(b, hasBoth ? '% de cada clase' : '% de puntos') +
      `<g class="cursor" style="display:none"><rect class="cur-band" x="0" y="${b.m.t}" width="${slot}" height="${b.ih}"/></g>` +
      `<rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cur = el.querySelector('.cur-band');
    hover(el, (px) => {
      const k = Math.floor((px - b.m.l) / slot);
      if (k < 0 || k >= nb) return null;
      cur.setAttribute('x', b.m.l + k * slot);
      let h = `<span class="tip-h">Score ${(lo + k * bw).toFixed(2)} – ${(lo + (k + 1) * bw).toFixed(2)}</span>`;
      h += row(dot('var(--s1)') + (hasBoth ? 'Normales' : 'Puntos'), `${c0[k]} (${(100 * share(c0[k], t0)).toFixed(1)}%)`);
      if (hasBoth) h += row(dot('var(--s2)') + 'Anómalos', `${c1[k]} (${(100 * share(c1[k], t1)).toFixed(1)}%)`);
      return h + '<span class="tip-sub">Cuanto menos se solapan las dos distribuciones, mejor separa el modelo.</span>';
    });
  };

  /* ---------- 2. Curvas ROC y PR ---------- */
  C.curve = function (el, { kind, c, op, baseline }) {
    const b = box(el, 0.98, 210, 300, { t: 12, r: 12, b: 36, l: 42 });
    const isRoc = kind === 'roc';
    const X = isRoc ? c.fpr : c.recall, Y = isRoc ? c.tpr : c.precision, T = isRoc ? c.thrRoc : c.thrPr;
    const x = lin(0, 1, b.m.l, b.m.l + b.iw), y = lin(0, 1, b.m.t + b.ih, b.m.t);
    let d = '';
    for (let i = 0; i < X.length; i++) d += (i ? 'L' : 'M') + x(X[i]).toFixed(1) + ' ' + y(Y[i]).toFixed(1);
    const base = isRoc
      ? `<line class="ref" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>`
      : `<line class="ref" x1="${x(0)}" y1="${y(baseline)}" x2="${x(1)}" y2="${y(baseline)}"/>`;
    const baseTxt = isRoc
      ? `<text class="mut" x="${x(0.55)}" y="${y(0.55) + 16}" text-anchor="start">azar</text>`
      : `<text class="mut" x="${x(1) - 4}" y="${y(baseline) - 5}" text-anchor="end">azar (${fmt(baseline, 2)})</text>`;
    const opX = isRoc ? op.fpr : op.recall, opY = isRoc ? op.tpr : op.precision;
    const ticks = [0, 0.25, 0.5, 0.75, 1];
    el.innerHTML = svgOpen(b, isRoc ? 'Curva ROC' : 'Curva Precisión-Recall') +
      gridY(b, y, ticks, (t) => t.toFixed(2)) + axisX(b, x, ticks, (t) => t.toFixed(2)) +
      `<path d="${d}L${x(X[X.length - 1])} ${y(isRoc ? 0 : 0)}L${x(0)} ${y(0)}Z" style="fill:var(--s1);opacity:.09"/>` +
      base + baseTxt + `<path d="${d}" fill="none" stroke-width="2" stroke-linejoin="round" style="stroke:var(--s1)"/>` +
      `<circle cx="${x(opX)}" cy="${y(opY)}" r="5.5" style="fill:var(--s2);stroke:var(--surface);stroke-width:2"/>` +
      xTitle(b, isRoc ? 'Tasa de falsos positivos' : 'Recall') + yTitle(b, isRoc ? 'Recall (TPR)' : 'Precisión') +
      `<g class="cursor" style="display:none"><circle class="cur-dot" r="4" style="fill:var(--surface);stroke:var(--ink);stroke-width:2"/></g>` +
      `<rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cd = el.querySelector('.cur-dot');
    hover(el, (px) => {
      const v = x.inv(px);
      let best = 0, bd = Infinity;
      for (let i = 0; i < X.length; i++) { const dd = Math.abs(X[i] - v); if (dd < bd) { bd = dd; best = i; } }
      cd.setAttribute('cx', x(X[best])); cd.setAttribute('cy', y(Y[best]));
      const s = T[best];
      return `<span class="tip-h">Si el umbral fuera ${Number.isFinite(s) ? fmt(s, 3) : '∞'}</span>` +
        (isRoc ? row('Falsos positivos (FPR)', fmt(X[best], 2)) + row('Recall (TPR)', fmt(Y[best], 2))
          : row('Recall', fmt(X[best], 2)) + row('Precisión', fmt(Y[best], 2))) +
        '<span class="tip-sub">Punto naranja = umbral actual (contamination).</span>';
    });
  };

  /* ---------- 3. Barrido de un hiperparámetro ---------- */
  C.sweep = function (el, { spec, results, metric, metricLabel, current, running, total, mark }) {
    const b = box(el, 0.5, 210, 300, { t: 18, r: 16, b: 44, l: 46 });
    if (!results.length) {
      el.innerHTML = `<div class="empty">Calculando el barrido de <code>${esc(spec.code)}</code>…</div>`;
      return;
    }
    let vmin = Infinity, vmax = -Infinity;
    for (const r of results) for (const v of r.metrics[metric]) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    const padY = Math.max(0.02, (vmax - vmin) * 0.15);
    const y0 = clamp(Math.floor((vmin - padY) * 20) / 20, 0, 0.95), y1 = clamp(Math.ceil((vmax + padY) * 20) / 20, y0 + 0.1, 1);
    const yy = lin(y0, y1, b.m.t + b.ih, b.m.t);
    const kind = spec.kind, vals = results.map((r) => r.x);
    let x;
    if (kind === 'log') x = logs(Math.min(...vals), Math.max(...vals), b.m.l + 8, b.m.l + b.iw - 8);
    else if (kind === 'lin') x = lin(Math.min(...vals), Math.max(...vals), b.m.l + 8, b.m.l + b.iw - 8);
    else { const k = vals.length; x = (v) => b.m.l + ((vals.indexOf(v) + 0.5) / k) * b.iw; }
    const stat = results.map((r) => {
      const a = r.metrics[metric], m = a.reduce((s, v) => s + v, 0) / a.length;
      return { m, sd: Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length), lo: Math.min(...a), hi: Math.max(...a) };
    });
    let body = '';
    if (kind !== 'cat') {
      const up = stat.map((s, i) => `${i ? 'L' : 'M'}${x(vals[i]).toFixed(1)} ${yy(s.hi).toFixed(1)}`).join('');
      const dn = stat.map((s, i) => `L${x(vals[stat.length - 1 - i]).toFixed(1)} ${yy(stat[stat.length - 1 - i].lo).toFixed(1)}`).join('');
      body += `<path d="${up}${dn}Z" style="fill:var(--s1);opacity:.14"/>`;
      results.forEach((r, i) => r.metrics[metric].forEach((v) => { body += `<circle cx="${x(vals[i]).toFixed(1)}" cy="${yy(v).toFixed(1)}" r="2" style="fill:var(--s1);opacity:.4"/>`; }));
      body += `<path d="${stat.map((s, i) => `${i ? 'L' : 'M'}${x(vals[i]).toFixed(1)} ${yy(s.m).toFixed(1)}`).join('')}" fill="none" stroke-width="2" stroke-linejoin="round" style="stroke:var(--s1)"/>`;
      stat.forEach((s, i) => { body += `<circle cx="${x(vals[i]).toFixed(1)}" cy="${yy(s.m).toFixed(1)}" r="3.6" style="fill:var(--s1);stroke:var(--surface);stroke-width:1.5"/>`; });
    } else {
      if (spec.band) {
        const all = results.map((r) => r.metrics[metric][0]);
        const m = all.reduce((s, v) => s + v, 0) / all.length, sd = Math.sqrt(all.reduce((s, v) => s + (v - m) ** 2, 0) / all.length);
        body += `<rect x="${b.m.l}" y="${yy(m + sd)}" width="${b.iw}" height="${Math.max(1, yy(m - sd) - yy(m + sd))}" style="fill:var(--s1);opacity:.12"/>` +
          `<line class="ref" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${yy(m)}" y2="${yy(m)}"/>` +
          `<text class="mut" x="${b.m.l + b.iw - 3}" y="${yy(m + sd) - 4}" text-anchor="end">media ${fmt(m, 3)} ± ${fmt(sd, 3)}</text>`;
      }
      results.forEach((r, i) => {
        const cx = x(vals[i]), s = stat[i];
        if (r.metrics[metric].length > 1) body += `<line x1="${cx}" x2="${cx}" y1="${yy(s.lo)}" y2="${yy(s.hi)}" stroke-width="10" stroke-linecap="round" style="stroke:var(--s1);opacity:.14"/>`;
        r.metrics[metric].forEach((v, j) => { body += `<circle cx="${(cx + (r.metrics[metric].length > 1 ? (j - (r.metrics[metric].length - 1) / 2) * 4 : 0)).toFixed(1)}" cy="${yy(v).toFixed(1)}" r="${r.metrics[metric].length > 1 ? 2.4 : 4.5}" style="fill:var(--s1);opacity:${r.metrics[metric].length > 1 ? 0.55 : 1};stroke:var(--surface);stroke-width:${r.metrics[metric].length > 1 ? 0 : 1.5}"/>`; });
        if (r.metrics[metric].length > 1) body += `<line x1="${cx - 9}" x2="${cx + 9}" y1="${yy(s.m)}" y2="${yy(s.m)}" stroke-width="3" stroke-linecap="round" style="stroke:var(--s1)"/>`;
      });
    }
    // valor actual
    let curLine = '';
    if (current != null && vals.includes(current)) {
      const cx = x(current);
      curLine = `<line class="thr" x1="${cx}" x2="${cx}" y1="${b.m.t - 4}" y2="${b.m.t + b.ih}"/><text class="ink" x="${cx}" y="${b.m.t - 7}" text-anchor="middle">actual</text>`;
    }
    if (mark && spec.kind === 'lin' && mark.x >= Math.min(...vals) && mark.x <= Math.max(...vals)) {
      const mx = x(mark.x);
      curLine += `<line class="ref" x1="${mx}" x2="${mx}" y1="${b.m.t}" y2="${b.m.t + b.ih}"/><text class="mut" x="${mx + 4}" y="${b.m.t + b.ih - 6}">${esc(mark.label)}</text>`;
    }
    const yt = niceTicks(y0, y1, 4);
    let xt = vals;
    if (vals.length > 7) { const st = Math.ceil(vals.length / (b.iw / 46)); xt = vals.filter((_, i) => i % st === 0 || vals[i] === current); }
    const xl = (v) => (spec.tickLabel ? spec.tickLabel(v) : v);
    let xa = `<line class="axis" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${b.m.t + b.ih}" y2="${b.m.t + b.ih}"/>`;
    xt.forEach((v) => { xa += `<line class="axis" x1="${x(v)}" x2="${x(v)}" y1="${b.m.t + b.ih}" y2="${b.m.t + b.ih + 4}"/><text x="${x(v)}" y="${b.m.t + b.ih + 17}" text-anchor="middle">${xl(v)}</text>`; });
    const note = running ? `<text class="mut" x="${b.m.l + b.iw}" y="${b.h - 4}" text-anchor="end">calculando ${results.length}/${total}…</text>` : '';
    el.innerHTML = svgOpen(b, `Barrido de ${spec.code}: ${metricLabel}`) + gridY(b, yy, yt, (t) => t.toFixed(2)) + xa + body + curLine +
      xTitle(b, esc(spec.axisLabel || spec.code)).replace(`x="${b.m.l + b.iw / 2}"`, `x="${b.m.l + b.iw / 2 - (running ? 40 : 0)}"`) + yTitle(b, esc(metricLabel)) + note +
      `<g class="cursor" style="display:none"><line class="cur-line" y1="${b.m.t}" y2="${b.m.t + b.ih}"/></g>` +
      `<rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cl = el.querySelector('.cur-line');
    hover(el, (px) => {
      let best = 0, bd = Infinity;
      vals.forEach((v, i) => { const dd = Math.abs(x(v) - px); if (dd < bd) { bd = dd; best = i; } });
      const cx = x(vals[best]);
      cl.setAttribute('x1', cx); cl.setAttribute('x2', cx);
      const s = stat[best], r = results[best];
      let h = `<span class="tip-h">${esc(spec.code)} = ${esc(String(xl(vals[best])))}</span>`;
      if (r.metrics[metric].length > 1) {
        h += row('media', fmt(s.m, 3)) + row('desv. estándar', fmt(s.sd, 3)) + row('rango', `${fmt(s.lo, 3)} – ${fmt(s.hi, 3)}`);
        h += `<span class="tip-sub">${r.metrics[metric].length} semillas distintas por valor</span>`;
      } else h += row(metricLabel, fmt(s.m, 3));
      return h;
    });
  };

  /* ---------- 4. Convergencia del score con el nº de árboles ---------- */
  C.convergence = function (el, { series, thr, T, cPsi }) {
    const b = box(el, 0.55, 210, 320, { t: 16, r: 104, b: 40, l: 46 });
    // score acumulado tras t árboles para cada serie
    const S = series.map((s) => {
      const out = new Float64Array(T); let acc = 0;
      for (let t = 0; t < T; t++) { acc += s.h[t]; out[t] = cPsi > 0 ? Math.pow(2, -(acc / (t + 1)) / cPsi) : 0.5; }
      return out;
    });
    let mn = thr, mx = thr;
    S.forEach((a) => a.forEach((v) => { if (v < mn) mn = v; if (v > mx) mx = v; }));
    const pad = Math.max(0.02, (mx - mn) * 0.08);
    const y0 = clamp(Math.floor((mn - pad) * 20) / 20, 0, 0.95), y1 = clamp(Math.ceil((mx + pad) * 20) / 20, y0 + 0.1, 1);
    const x = T > 1 ? logs(1, T, b.m.l + 4, b.m.l + b.iw) : () => b.m.l + b.iw / 2;
    const yy = lin(y0, y1, b.m.t + b.ih, b.m.t);
    let paths = '', labels = '';
    const lab = series.map((s, i) => ({ s, i, y: yy(S[i][T - 1]) })).sort((a, c) => a.y - c.y);
    for (let k = 1; k < lab.length; k++) if (lab[k].y - lab[k - 1].y < 13) lab[k].y = lab[k - 1].y + 13;
    series.forEach((s, i) => {
      const d = Array.from(S[i], (v, t) => `${t ? 'L' : 'M'}${x(t + 1).toFixed(1)} ${yy(v).toFixed(1)}`).join('');
      paths += `<path d="${d}" fill="none" stroke-width="${s.dash ? 2 : 2}" stroke-linejoin="round" ${s.dash ? 'stroke-dasharray="5 3"' : ''} style="stroke:${s.color}"/>`;
      paths += `<circle cx="${x(T)}" cy="${yy(S[i][T - 1])}" r="3.5" style="fill:${s.color};stroke:var(--surface);stroke-width:1.5"/>`;
    });
    lab.forEach((l) => { labels += `<text class="ink" x="${b.m.l + b.iw + 9}" y="${l.y + 4}">${esc(l.s.name)}</text>`; });
    const thrLine = thr >= y0 && thr <= y1
      ? `<line class="thr" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${yy(thr)}" y2="${yy(thr)}"/><text class="mut" x="${b.m.l + 6}" y="${yy(thr) - 5}">umbral ${fmt(thr, 2)}</text>` : '';
    const xt = [1, 2, 5, 10, 20, 50, 100, 200, 500].filter((v) => v <= T);
    if (!xt.includes(T) && T > 1 && Math.log(T / xt[xt.length - 1]) > 0.35) xt.push(T);
    el.innerHTML = svgOpen(b, 'Convergencia del score según el número de árboles') +
      gridY(b, yy, niceTicks(y0, y1, 4), (t) => t.toFixed(2)) + axisX(b, x, xt) + thrLine + paths + labels +
      xTitle(b, 'árboles promediados (escala log)') + yTitle(b, 'score s(x)') +
      `<g class="cursor" style="display:none"><line class="cur-line" y1="${b.m.t}" y2="${b.m.t + b.ih}"/></g>` +
      `<rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cl = el.querySelector('.cur-line');
    hover(el, (px) => {
      const t = T > 1 ? clamp(Math.round(x.inv(px)), 1, T) : 1;
      cl.setAttribute('x1', x(t)); cl.setAttribute('x2', x(t));
      let h = `<span class="tip-h">Tras ${t} árbol${t > 1 ? 'es' : ''}</span>`;
      series.forEach((s, i) => { h += row(dot(s.color) + esc(s.name), fmt(S[i][t - 1], 3)); });
      return h;
    });
  };

  /* ---------- 5. Anatomía: distribución de h_t(x) entre árboles ---------- */
  C.pathHist = function (el, { h, mean, cPsi }) {
    const b = box(el, 0.5, 180, 260, { t: 20, r: 14, b: 36, l: 40 });
    let mx = cPsi;
    h.forEach((v) => { if (v > mx) mx = v; });
    const hi = Math.ceil(mx + 0.5), nb = hi;
    const counts = new Array(nb).fill(0);
    h.forEach((v) => { counts[clamp(Math.floor(v), 0, nb - 1)]++; });
    const cmax = Math.max(...counts);
    const x = lin(0, hi, b.m.l, b.m.l + b.iw), yy = lin(0, cmax * 1.08, b.m.t + b.ih, b.m.t);
    const slot = b.iw / nb;
    const bars = counts.map((c, k) => `<rect x="${b.m.l + k * slot + 1}" y="${yy(c)}" width="${Math.max(1, slot - 2)}" height="${b.m.t + b.ih - yy(c)}" rx="1.5" style="fill:var(--s1)"/>`).join('');
    const meanRight = x(mean) < b.m.l + b.iw - 80;
    el.innerHTML = svgOpen(b, 'Distribución de la profundidad de aislamiento por árbol') +
      gridY(b, yy, niceTicks(0, cmax, 3)) + axisX(b, x, Array.from({ length: hi + 1 }, (_, i) => i).filter((v) => hi < 14 || v % 2 === 0)) + bars +
      `<line class="ref" x1="${x(cPsi)}" x2="${x(cPsi)}" y1="${b.m.t}" y2="${b.m.t + b.ih}"/><text class="mut" x="${x(cPsi) + 4}" y="${b.m.t + 11}">c(ψ)</text>` +
      `<line class="thr" x1="${x(mean)}" x2="${x(mean)}" y1="${b.m.t - 6}" y2="${b.m.t + b.ih}"/><text class="ink" x="${x(mean) + (meanRight ? 5 : -5)}" y="${b.m.t - 9}" text-anchor="${meanRight ? 'start' : 'end'}">E[h] = ${fmt(mean, 2)}</text>` +
      xTitle(b, 'profundidad de aislamiento h(x) en cada árbol') + yTitle(b, 'nº de árboles') +
      `<g class="cursor" style="display:none"><rect class="cur-band" x="0" y="${b.m.t}" width="${slot}" height="${b.ih}"/></g>` +
      `<rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cb = el.querySelector('.cur-band');
    hover(el, (px) => {
      const k = Math.floor((px - b.m.l) / slot);
      if (k < 0 || k >= nb) return null;
      cb.setAttribute('x', b.m.l + k * slot);
      return `<span class="tip-h">h entre ${k} y ${k + 1}</span>${row('árboles', counts[k])}<span class="tip-sub">Cada árbol da una opinión distinta; el score usa el promedio.</span>`;
    });
  };
})();
