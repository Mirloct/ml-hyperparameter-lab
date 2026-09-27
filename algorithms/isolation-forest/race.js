/* Capítulo 1 — "Carrera de aislamiento": ¿cuántos cortes aleatorios hacen falta para dejar solo a un punto? */
(function () {
  'use strict';
  const ML = window.MLLab;
  const { $, $$, clamp, fmt } = ML;
  const IF = ML.IsolationForest, D = ML.datasets;
  const canvas = $('#raceCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const N = 90;
  const spread = (d) => { for (let i = 0; i < d.n; i++) { d.X[i * 2] = clamp(0.5 + (d.X[i * 2] - 0.5) * 2.1, 0.04, 0.96); d.X[i * 2 + 1] = clamp(0.5 + (d.X[i * 2 + 1] - 0.5) * 2.1, 0.04, 0.96); } return d; };
  let data = spread(D.generate('blob', N, 0.04, 21));
  const S = { target: -1, cuts: [], size: 0, playing: false, hist: { a: [], n: [] }, rng: IF.mulberry32(12345), auto: null };
  let anomIdx = 0, normIdx = 0, W = 0, dpr = 1;

  function pickTargets() {
    let far = -1, fd = -1, near = -1, nd = Infinity;
    for (let i = 0; i < N; i++) {
      const d = Math.hypot(data.X[i * 2] - 0.5, data.X[i * 2 + 1] - 0.5);
      if (d > fd) { fd = d; far = i; }
      if (d < nd) { nd = d; near = i; }
    }
    anomIdx = far; normIdx = near;
  }
  pickTargets();

  /** Un corte aleatorio sobre la celda actual (como un iTree: variable y valor al azar dentro del rango de los puntos de la celda). */
  function cell(cuts) {
    let r = [0, 0, 1, 1]; // x0,y0,x1,y1
    const t = S.target, tx = data.X[t * 2], ty = data.X[t * 2 + 1];
    cuts.forEach((c) => {
      if (c.f === 0) { if (tx < c.v) r[2] = c.v; else r[0] = c.v; } else if (ty < c.v) r[3] = c.v; else r[1] = c.v;
    });
    return r;
  }
  function inside(i, r) { const x = data.X[i * 2], y = data.X[i * 2 + 1]; return x >= r[0] && x < r[2] && y >= r[1] && y < r[3]; }
  function members(r) { const m = []; for (let i = 0; i < N; i++) if (inside(i, r)) m.push(i); return m; }

  function oneCut(cuts, rng) {
    const r = cell(cuts), m = members(r);
    if (m.length <= 1) return null;
    for (let tries = 0; tries < 8; tries++) {
      const f = rng() < 0.5 ? 0 : 1;
      let lo = Infinity, hi = -Infinity;
      m.forEach((i) => { const v = data.X[i * 2 + f]; if (v < lo) lo = v; if (v > hi) hi = v; });
      if (hi > lo) return { f, v: lo + rng() * (hi - lo) };
    }
    return null;
  }
  function cutsToIsolate(target, rng) {
    const save = S.target; S.target = target; const cuts = [];
    for (let g = 0; g < 60; g++) { const c = oneCut(cuts, rng); if (!c) break; cuts.push(c); }
    S.target = save; return cuts.length;
  }

  function size() {
    const r = canvas.getBoundingClientRect(); dpr = Math.min(devicePixelRatio || 1, 2);
    const nw = Math.max(240, Math.round(r.width));
    if (nw === W && canvas.width) return false;
    W = nw; canvas.width = W * dpr; canvas.height = W * dpr; return true;
  }

  function draw() {
    if (!W) size();
    const c = ML.colors();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = c.surface; ctx.fillRect(0, 0, W, W);
    const P = (x) => x * W, Q = (y) => (1 - y) * W;
    const t = S.target;
    if (t >= 0) {
      const r = cell(S.cuts);
      ctx.fillStyle = c.dark ? 'rgba(57,135,229,0.28)' : 'rgba(42,120,214,0.16)';
      ctx.fillRect(P(r[0]), Q(r[3]), P(r[2]) - P(r[0]), Q(r[1]) - Q(r[3]));
      // cortes: se dibujan acotados a la celda vigente en el momento del corte
      let box = [0, 0, 1, 1];
      S.cuts.forEach((cut) => {
        ctx.beginPath();
        if (cut.f === 0) { ctx.moveTo(P(cut.v), Q(box[1])); ctx.lineTo(P(cut.v), Q(box[3])); } else { ctx.moveTo(P(box[0]), Q(cut.v)); ctx.lineTo(P(box[2]), Q(cut.v)); }
        ctx.strokeStyle = c.ink; ctx.lineWidth = 1.6; ctx.stroke();
        const tx = data.X[t * 2], ty = data.X[t * 2 + 1];
        if (cut.f === 0) { if (tx < cut.v) box[2] = cut.v; else box[0] = cut.v; } else if (ty < cut.v) box[3] = cut.v; else box[1] = cut.v;
      });
      ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.strokeRect(P(r[0]), Q(r[3]), P(r[2]) - P(r[0]), Q(r[1]) - Q(r[3]));
    }
    const inCell = t >= 0 ? members(cell(S.cuts)) : [];
    for (let i = 0; i < N; i++) {
      const x = P(data.X[i * 2]), y = Q(data.X[i * 2 + 1]), isT = i === t, alive = t < 0 || inCell.includes(i);
      ctx.globalAlpha = alive ? 1 : 0.25;
      ctx.beginPath(); ctx.arc(x, y, isT ? 6 : 3.4, 0, 6.2832);
      ctx.fillStyle = isT ? (i === anomIdx ? c.s2 : c.s1) : c.ink2; ctx.fill();
      ctx.lineWidth = 1.4; ctx.strokeStyle = c.surface; ctx.stroke();
    }
    ctx.globalAlpha = 1;
    [anomIdx, normIdx].forEach((i) => {
      const x = P(data.X[i * 2]), y = Q(data.X[i * 2 + 1]);
      ctx.beginPath(); ctx.arc(x, y, i === t ? 11 : 9, 0, 6.2832); ctx.lineWidth = 1.8; ctx.strokeStyle = i === anomIdx ? c.s2 : c.s1; ctx.stroke();
    });
  }

  function status() {
    const t = S.target, el = $('#raceStatus');
    if (t < 0) { el.innerHTML = 'Elige un punto <b>(clic)</b> o usa los botones de arriba y pulsa <b>Hacer un corte</b>.'; return; }
    const m = members(cell(S.cuts)).length, name = t === anomIdx ? 'el punto anómalo' : t === normIdx ? 'el punto normal' : 'el punto elegido';
    if (m <= 1) el.innerHTML = `<b>¡Aislado!</b> Hicieron falta <b>${S.cuts.length}</b> cortes para dejar solo a ${name}.`;
    else el.innerHTML = `Cortes: <b>${S.cuts.length}</b> · en la celda de ${name} quedan <b>${m}</b> de ${N} puntos.`;
    $('#raceCuts').textContent = S.cuts.length;
  }

  function setTarget(i) { S.target = i; S.cuts = []; draw(); status(); $$('#raceTarget button').forEach((b) => b.setAttribute('aria-pressed', String((b.dataset.t === 'a' && i === anomIdx) || (b.dataset.t === 'n' && i === normIdx)))); }
  function step() { if (S.target < 0) setTarget(anomIdx); const c = oneCut(S.cuts, S.rng); if (c) S.cuts.push(c); else stop(); draw(); status(); if (!c) stop(); }
  function stop() { if (S.auto) { clearInterval(S.auto); S.auto = null; } $('#racePlay').textContent = '▶ Automático'; }

  /** Repite la carrera muchas veces para ambos puntos y dibuja las distribuciones de nº de cortes. */
  function many() {
    const rng = IF.mulberry32((Math.random() * 1e9) | 0), R = 300, a = [], n = [];
    for (let r = 0; r < R; r++) { a.push(cutsToIsolate(anomIdx, rng)); n.push(cutsToIsolate(normIdx, rng)); }
    S.hist = { a, n }; drawHist();
  }
  function drawHist() {
    const host = $('#raceHist'), { a, n } = S.hist;
    if (!a.length) { host.innerHTML = '<div class="empty">Pulsa «Repetir 300 veces» para ver cuántos cortes necesita cada punto <b>en promedio</b>.</div>'; return; }
    const H = ML.charts._h, b = H.box(host, 0.5, 170, 230, { t: 22, r: 12, b: 38, l: 40 });
    const mx = Math.max(...a, ...n) + 1, bins = mx + 1;
    const ca = new Array(bins).fill(0), cn = new Array(bins).fill(0);
    a.forEach((v) => ca[v]++); n.forEach((v) => cn[v]++);
    const ymax = Math.max(...ca, ...cn), x = H.lin(0, bins, b.m.l, b.m.l + b.iw), y = H.lin(0, ymax * 1.1, b.m.t + b.ih, b.m.t), sl = b.iw / bins;
    const meanA = a.reduce((s, v) => s + v, 0) / a.length, meanN = n.reduce((s, v) => s + v, 0) / n.length;
    let bars = '';
    for (let k = 0; k < bins; k++) {
      const x0 = b.m.l + k * sl + sl / 2, w = Math.max(2, sl * 0.4);
      bars += `<rect x="${x0 - w - 0.5}" y="${y(cn[k])}" width="${w}" height="${b.m.t + b.ih - y(cn[k])}" rx="1.5" style="fill:var(--s1)"/><rect x="${x0 + 0.5}" y="${y(ca[k])}" width="${w}" height="${b.m.t + b.ih - y(ca[k])}" rx="1.5" style="fill:var(--s2)"/>`;
    }
    const lines = `<line class="thr" x1="${x(meanA + 0.5)}" x2="${x(meanA + 0.5)}" y1="${b.m.t}" y2="${b.m.t + b.ih}" style="stroke:var(--s2)"/><line class="thr" x1="${x(meanN + 0.5)}" x2="${x(meanN + 0.5)}" y1="${b.m.t}" y2="${b.m.t + b.ih}" style="stroke:var(--s1)"/>`;
    host.innerHTML = H.svgOpen(b, 'Distribución del número de cortes para aislar cada punto') + H.gridY(b, y, H.niceTicks(0, ymax, 3)) +
      H.axisX(b, (v) => x(v + 0.5), Array.from({ length: bins }, (_, i) => i).filter((v) => bins < 16 || v % 2 === 0)) + bars + lines +
      H.xTitle(b, 'cortes necesarios para aislar el punto') + H.yTitle(b, 'de 300 carreras') + '</svg>';
    $('#raceMeans').innerHTML = `Promedio: <span class="dotc" style="background:var(--s2)"></span>anómalo <b>${meanA.toFixed(1)}</b> cortes · <span class="dotc" style="background:var(--s1)"></span>normal <b>${meanN.toFixed(1)}</b> cortes. Ese promedio es <b>E[h(x)]</b>: menor ⇒ más anómalo.`;
  }

  function init() {
    size(); setTarget(-1); draw(); drawHist();
    $('#raceCut').addEventListener('click', step);
    $('#racePlay').addEventListener('click', () => {
      if (S.auto) { stop(); return; }
      if (S.target < 0 || members(cell(S.cuts)).length <= 1) setTarget(S.target < 0 ? anomIdx : S.target);
      $('#racePlay').textContent = '⏸ Pausa'; S.auto = setInterval(() => { step(); if (members(cell(S.cuts)).length <= 1) stop(); }, 650);
    });
    $('#raceReset').addEventListener('click', () => { stop(); setTarget(S.target); });
    $('#raceMany').addEventListener('click', many);
    $$('#raceTarget button').forEach((b) => b.addEventListener('click', () => { stop(); setTarget(b.dataset.t === 'a' ? anomIdx : normIdx); }));
    $('#raceNew').addEventListener('click', () => { stop(); data = spread(D.generate('blob', N, 0.04, Math.floor(Math.random() * 1000))); pickTargets(); S.hist = { a: [], n: [] }; setTarget(-1); drawHist(); });
    canvas.addEventListener('pointerdown', (e) => {
      const r = canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      let best = -1, bd = 22 * 22;
      for (let i = 0; i < N; i++) { const d = (data.X[i * 2] * W - px) ** 2 + ((1 - data.X[i * 2 + 1]) * W - py) ** 2; if (d < bd) { bd = d; best = i; } }
      if (best >= 0) { stop(); setTarget(best); }
    });
    window.addEventListener('themechange', draw);
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (size()) { draw(); drawHist(); } }, 120); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
