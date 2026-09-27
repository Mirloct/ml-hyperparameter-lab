/* Controlador del laboratorio de Isolation Forest: estado, entrenamiento en vivo y render. */
(function () {
  'use strict';
  const ML = window.MLLab;
  const IF = ML.IsolationForest, M = ML.metrics, D = ML.datasets, Ch = ML.charts;
  const { $, $$, clamp, fmt, pct, esc } = ML;
  const PARAMS = ML.IF_PARAMS, PMAP = Object.fromEntries(PARAMS.map((p) => [p.id, p]));

  const DEFAULTS = { nEstimators: 100, maxSamples: 256, contamination: 'auto', maxFeatures: 1, bootstrap: false, seed: 42 };
  const STOPS = {
    nEstimators: [1, 2, 3, 5, 10, 20, 50, 100, 150, 200, 300],
    maxSamples: [4, 8, 16, 32, 64, 128, 256, 512],
  };
  const SWEEPS = {
    nEstimators: { code: 'n_estimators', kind: 'log', values: [1, 2, 3, 5, 8, 12, 20, 35, 50, 75, 100, 150, 200, 300], axisLabel: 'n_estimators (nº de árboles, escala log)', reps: 6 },
    maxSamples: { code: 'max_samples', kind: 'log', values: [4, 8, 16, 32, 64, 128, 256, 512], axisLabel: 'max_samples ψ (escala log)', reps: 6 },
    contamination: { code: 'contamination', kind: 'lin', values: [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3], tickLabel: (v) => Math.round(v * 100) + '%', axisLabel: 'contamination (fracción marcada como anómala)', reps: 4 },
    maxFeatures: { code: 'max_features', kind: 'cat', values: [0.5, 1], tickLabel: (v) => (v === 0.5 ? '0.5 · 1 de 2' : '1.0 · 2 de 2'), axisLabel: 'max_features', reps: 8 },
    bootstrap: { code: 'bootstrap', kind: 'cat', values: [false, true], tickLabel: (v) => (v ? 'True' : 'False'), axisLabel: 'bootstrap', reps: 8 },
    seed: { code: 'random_state', kind: 'cat', values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], band: true, axisLabel: 'random_state (cada punto = una semilla distinta)', reps: 1 },
  };
  const SWEEP_HINT = {
    nEstimators: 'Cada valor se entrena con 6 semillas distintas. La banda se estrecha al subir T: menos varianza.',
    maxSamples: 'Compara: no hay ψ universal; el óptimo depende de los datos (prueba "Anomalías agrupadas" y "Dos lunas").',
    contamination: 'Con ROC-AUC o AP verías una línea plana: contamination solo mueve el umbral, y por eso afecta al F1.',
    maxFeatures: 'Con 2 columnas solo hay dos opciones. Prueba el dataset "Anillo con hueco".',
    bootstrap: 'Compara las dos categorías: la diferencia suele ser pequeña frente a la variabilidad entre semillas.',
    seed: 'Cada punto es el mismo modelo con otra semilla. La dispersión es la varianza atribuible solo al azar.',
  };
  const DEFAULT_SWEEP_METRIC = { contamination: 'f1' };

  const state = {
    dataset: 'blob', dataSeed: 11, nPoints: 500, anomalyFrac: 0.06,
    added: [], params: { ...DEFAULTS }, active: 'nEstimators', open: 'nEstimators',
    view: 'map', treeIdx: 0, depthShown: 3, selected: -1, hover: -1, addMode: 'select',
    showHeat: true, showLabels: true, baseline: null, sweepMetric: 'ap', lastChange: null, lesson: null,
  };
  let data = null, model = null, dataVersion = 0;

  /* =====================================================================
   *  Datos y modelo
   * ===================================================================== */
  function buildData() {
    const base = D.generate(state.dataset, state.nPoints, state.anomalyFrac, state.dataSeed);
    const k = state.added.length, n = base.n + k;
    const X = new Float64Array(n * 2), y = new Uint8Array(n);
    X.set(base.X); y.set(base.y);
    state.added.forEach((p, i) => { X[(base.n + i) * 2] = p.x; X[(base.n + i) * 2 + 1] = p.y; y[base.n + i] = p.label; });
    data = { X, y, n, nBase: base.n };
    dataVersion++;
    scoreCache.clear();
  }

  let gridXY = null, gridG = 0;
  function ensureGrid(G) {
    if (gridG === G) return;
    gridG = G; gridXY = new Float64Array(G * G * 2);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      gridXY[(j * G + i) * 2] = i / (G - 1);
      gridXY[(j * G + i) * 2 + 1] = 1 - j / (G - 1);
    }
  }

  function compute() {
    const { X, y, n } = data;
    const f = IF.fit(X, n, 2, state.params);
    const scores = f.scores(X, n);
    const thr = IF.threshold(scores, state.params.contamination);
    const conf = M.confusion(scores, y, thr);
    const cv = M.curves(scores, y);
    const G = f.nEstimators >= 200 ? 56 : 68;
    ensureGrid(G);
    const grid = f.scores(gridXY, G * G);
    let pos = 0; for (let i = 0; i < n; i++) pos += y[i];
    model = { f, scores, thr, conf, cv, grid, G, n, pos };
    if (state.selected < 0 || state.selected >= n) state.selected = order(scores)[0];
    if (state.treeIdx >= f.nEstimators) state.treeIdx = 0;
    treeCache = null;
  }
  const order = (s) => Array.from(s, (_, i) => i).sort((a, b) => s[b] - s[a]);
  const ehOf = (s) => (model.f.cPsi > 0 ? -Math.log2(s) * model.f.cPsi : 0);

  function metricsNow() {
    const c = model.conf, cv = model.cv;
    return { auc: cv ? cv.auc : NaN, ap: cv ? cv.ap : NaN, precision: cv ? c.precision : NaN, recall: cv ? c.recall : NaN, f1: cv ? c.f1 : NaN, flagged: c.flagged, n: model.n };
  }

  /* =====================================================================
   *  Parámetros: cambios y tarjetas
   * ===================================================================== */
  function fmtParam(id, v = state.params[id]) {
    switch (id) {
      case 'contamination': return v === 'auto' ? 'auto' : (v * 100).toFixed(1) + '%';
      case 'maxFeatures': return Number(v).toFixed(1);
      case 'bootstrap': return v ? 'True' : 'False';
      default: return String(v);
    }
  }
  function setParam(id, value) {
    const old = state.params[id];
    if (old === value) return;
    const lc = state.lastChange;
    if (lc && lc.id === id && Date.now() - lc.t < 1800) { lc.to = value; lc.t = Date.now(); }
    else state.lastChange = { id, from: old, to: value, before: model ? metricsNow() : null, t: Date.now() };
    state.params[id] = value;
    openParam(id, false);
    scheduleUpdate();
  }
  function openParam(id, toggle) {
    const changed = state.active !== id;
    state.active = id;
    state.open = toggle && state.open === id ? null : id;
    $$('.pcard').forEach((el) => el.classList.toggle('active', el.dataset.id === state.open));
    $$('.pcard .p-head').forEach((b) => b.setAttribute('aria-expanded', String(b.closest('.pcard').dataset.id === state.open)));
    if (changed) { state.sweepMetric = DEFAULT_SWEEP_METRIC[id] || 'ap'; scheduleSweep(); renderSweepChrome(); }
  }

  function bodyHTML(p) {
    return `<p class="p-short">${p.short}</p><dl>
      <dt>¿Qué es?</dt><dd>${p.what}</dd>
      <dt>¿Qué controla en el modelo?</dt><dd>${p.controls}</dd>
      <dt>${p.upLabel || 'Si lo subes'}</dt><dd class="up">${p.up}</dd>
      <dt>${p.downLabel || 'Si lo bajas'}</dt><dd class="down">${p.down}</dd>
      <dt>Valor típico</dt><dd>${p.typical}</dd>
      <dt>Dónde se nota</dt><dd>${p.watch}</dd></dl>
      <div class="p-live" data-live></div>
      <button class="btn sm p-go" type="button" data-go>Ver dónde se nota ↓</button>`;
  }
  function controlHTML(p) {
    switch (p.id) {
      case 'nEstimators': case 'maxSamples': {
        const s = STOPS[p.id];
        return `<input type="range" min="0" max="${s.length - 1}" step="1" aria-label="${p.code}"><div class="p-ticks"><span>${s[0]}</span><span>${s[s.length - 1]}</span></div>`;
      }
      case 'contamination':
        return `<div class="contam-row"><label><input type="checkbox" data-auto> 'auto' (umbral fijo 0.5)</label></div>
          <input type="range" min="1" max="30" step="0.5" aria-label="contamination"><div class="p-ticks"><span>1%</span><span>30%</span></div>`;
      case 'maxFeatures':
        return `<div class="seg" role="group" aria-label="max_features"><button type="button" data-v="0.5">0.5 · 1 de 2</button><button type="button" data-v="1">1.0 · 2 de 2</button></div>`;
      case 'bootstrap':
        return `<div class="contam-row"><label><input type="checkbox" data-boot> Con reemplazo (<code>True</code>)</label></div>`;
      case 'seed':
        return `<div class="seed-row"><input type="number" min="0" max="99999" step="1" aria-label="random_state"><button class="btn sm" type="button" data-dice data-tip="Elige una semilla aleatoria nueva.">🎲 Nueva semilla</button></div>`;
    }
    return '';
  }

  function buildParamCards() {
    const host = $('#paramList');
    host.innerHTML = '';
    PARAMS.forEach((p) => {
      const el = document.createElement('div');
      el.className = 'pcard'; el.dataset.id = p.id;
      const tip = `<span class="tip-h">${esc(p.title)} · <code>${p.code}</code></span>${p.short}<span class="tip-sub">Clic en el encabezado para ver la explicación completa</span>`;
      el.innerHTML = `<button class="p-head" type="button" aria-expanded="false" data-tip="${esc(tip)}">
          <span class="p-sym" aria-hidden="true">${p.symbol}</span>
          <span class="p-title"><b>${p.title}</b><code>${p.code}</code></span>
          <span class="p-val" data-val></span><span class="p-chev" aria-hidden="true">▾</span></button>
        <div class="p-ctl" data-tip="${esc(tip)}">${controlHTML(p)}</div>
        <div class="p-body">${bodyHTML(p)}</div>`;
      host.appendChild(el);
      el.querySelector('.p-head').addEventListener('click', () => openParam(p.id, true));
      el.querySelector('[data-go]').addEventListener('click', () => scrollToCard(p.target));
      el.addEventListener('focusin', () => { if (state.active !== p.id) openParam(p.id, false); });
      wireControl(el, p);
    });
  }

  function wireControl(el, p) {
    const id = p.id;
    if (STOPS[id]) {
      el.querySelector('input[type=range]').addEventListener('input', (e) => setParam(id, STOPS[id][+e.target.value]));
    } else if (id === 'contamination') {
      const auto = el.querySelector('[data-auto]'), rng = el.querySelector('input[type=range]');
      auto.addEventListener('change', () => setParam(id, auto.checked ? 'auto' : (state.lastContam || 0.1)));
      rng.addEventListener('input', () => { state.lastContam = +rng.value / 100; setParam(id, state.lastContam); });
    } else if (id === 'maxFeatures') {
      $$('[data-v]', el).forEach((b) => b.addEventListener('click', () => setParam(id, +b.dataset.v)));
    } else if (id === 'bootstrap') {
      el.querySelector('[data-boot]').addEventListener('change', (e) => setParam(id, e.target.checked));
    } else if (id === 'seed') {
      const inp = el.querySelector('input');
      inp.addEventListener('input', () => { const v = Math.max(0, Math.floor(+inp.value || 0)); setParam(id, v); });
      el.querySelector('[data-dice]').addEventListener('click', () => { const v = Math.floor(Math.random() * 9999); inp.value = v; setParam(id, v); });
    }
  }

  function syncParamCards() {
    const P = state.params;
    PARAMS.forEach((p) => {
      const el = $(`.pcard[data-id="${p.id}"]`);
      el.querySelector('[data-val]').textContent = fmtParam(p.id);
      if (STOPS[p.id]) {
        const i = STOPS[p.id].indexOf(P[p.id]);
        el.querySelector('input[type=range]').value = i >= 0 ? i : STOPS[p.id].length - 1;
      } else if (p.id === 'contamination') {
        el.querySelector('[data-auto]').checked = P.contamination === 'auto';
        const r = el.querySelector('input[type=range]');
        r.disabled = P.contamination === 'auto';
        r.value = P.contamination === 'auto' ? Math.round((state.lastContam || 0.1) * 100) : P.contamination * 100;
      } else if (p.id === 'maxFeatures') {
        $$('[data-v]', el).forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.v === P.maxFeatures)));
      } else if (p.id === 'bootstrap') {
        el.querySelector('[data-boot]').checked = P.bootstrap;
      } else if (p.id === 'seed') {
        const inp = el.querySelector('input'); if (document.activeElement !== inp) inp.value = P.seed;
      }
      const ctx = { thr: model.thr, flagged: model.conf.flagged, n: model.n, pos: model.pos };
      el.querySelector('[data-live]').innerHTML = p.live(state, model.f, ctx);
    });
    $$('.pcard').forEach((el) => el.classList.toggle('active', el.dataset.id === state.open));
  }

  /* =====================================================================
   *  Datos: controles
   * ===================================================================== */
  function buildDataControls() {
    const list = $('#dsList');
    Object.entries(D.DEFS).forEach(([id, def]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.id = id; b.textContent = def.name; b.setAttribute('role', 'radio');
      b.addEventListener('click', () => setDataset(id));
      list.appendChild(b);
    });
    $('#fracRange').addEventListener('input', (e) => {
      state.anomalyFrac = +e.target.value / 100; $('#fracVal').textContent = e.target.value + '%';
      state.added = []; state.selected = -1; state.lastChange = null; scheduleRebuild();
    });
    $('#nSel').addEventListener('change', (e) => { state.nPoints = +e.target.value; state.added = []; state.selected = -1; scheduleRebuild(); });
    $('#resampleBtn').addEventListener('click', () => { state.dataSeed++; state.added = []; state.selected = -1; scheduleRebuild(); });
    $('#undoBtn').addEventListener('click', () => { state.added = []; state.selected = -1; scheduleRebuild(); });
  }
  function setDataset(id) {
    state.dataset = id; state.added = []; state.selected = -1; state.lastChange = null; state.baseline = null;
    scheduleRebuild();
  }
  function syncDataControls() {
    $$('#dsList button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.id === state.dataset)));
    $('#dsBlurb').textContent = D.DEFS[state.dataset].blurb;
    $('#fracRange').value = state.anomalyFrac * 100; $('#fracVal').textContent = (state.anomalyFrac * 100).toFixed(state.anomalyFrac * 100 % 1 ? 1 : 0) + '%';
    $('#nSel').value = String(state.nPoints);
    $('#undoBtn').disabled = !state.added.length;
  }

  /* =====================================================================
   *  Scheduler
   * ===================================================================== */
  let raf = 0, needRebuild = false;
  function scheduleUpdate() { if (raf) return; raf = requestAnimationFrame(() => { raf = 0; runUpdate(); }); }
  function scheduleRebuild() { needRebuild = true; scheduleUpdate(); }
  function runUpdate() {
    if (needRebuild || !data) { buildData(); needRebuild = false; }
    compute();
    renderAll();
    scheduleSweep();
    writeHash();
  }
  const scheduleSweep = ML.debounce(() => runSweep(), 300);

  /* =====================================================================
   *  Render: mapa (canvas)
   * ===================================================================== */
  const canvas = $('#mapCanvas'), ctx2d = canvas.getContext('2d');
  canvas.setAttribute('data-chart-tip', '');
  let W = 0, dpr = 1, treeCache = null;
  const heatCv = document.createElement('canvas');

  /** Redimensiona solo si cambió el tamaño (asignar canvas.width borra el lienzo). */
  function resizeCanvas() {
    const r = canvas.getBoundingClientRect();
    const nd = Math.min(window.devicePixelRatio || 1, 2.5), nw = Math.max(200, Math.round(r.width));
    if (nw === W && nd === dpr && canvas.width) return false;
    dpr = nd; W = nw;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(W * dpr);
    return true;
  }
  function heatLUT(dark) {
    const stops = dark ? [[26, 26, 25], [24, 79, 149], [42, 120, 214], [134, 182, 239]] : [[252, 252, 251], [205, 226, 251], [134, 182, 239], [57, 135, 229]];
    const lut = new Array(256);
    for (let k = 0; k < 256; k++) {
      const t = (k / 255) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
      lut[k] = stops[i].map((v, c) => Math.round(v + (stops[i + 1][c] - v) * f));
    }
    return lut;
  }
  const S_LO = 0.30, S_HI = 0.85;

  function drawHeat(c) {
    const G = model.G, lut = heatLUT(c.dark);
    heatCv.width = G; heatCv.height = G;
    const hc = heatCv.getContext('2d'), img = hc.createImageData(G, G);
    for (let k = 0; k < G * G; k++) {
      const t = clamp((model.grid[k] - S_LO) / (S_HI - S_LO), 0, 1), col = lut[Math.round(t * 255)];
      img.data[k * 4] = col[0]; img.data[k * 4 + 1] = col[1]; img.data[k * 4 + 2] = col[2]; img.data[k * 4 + 3] = 255;
    }
    hc.putImageData(img, 0, 0);
    const cell = W / (G - 1);
    ctx2d.imageSmoothingEnabled = true; ctx2d.imageSmoothingQuality = 'high';
    ctx2d.drawImage(heatCv, -cell / 2, -cell / 2, G * cell, G * cell);
    // barra de color
    const stops = Array.from({ length: 9 }, (_, k) => { const col = lut[Math.round((k / 8) * 255)]; return `rgb(${col.join(',')}) ${k * 12.5}%`; });
    $('#colorbar').style.background = `linear-gradient(90deg, ${stops.join(',')})`;
  }

  /** Contorno del umbral (marching squares con interpolación lineal). */
  function drawContour(c) {
    const { grid, G, thr } = model, cell = W / (G - 1);
    ctx2d.beginPath();
    for (let j = 0; j < G - 1; j++) for (let i = 0; i < G - 1; i++) {
      const tl = grid[j * G + i] - thr, tr = grid[j * G + i + 1] - thr, br = grid[(j + 1) * G + i + 1] - thr, bl = grid[(j + 1) * G + i] - thr;
      const idx = (tl > 0 ? 8 : 0) | (tr > 0 ? 4 : 0) | (br > 0 ? 2 : 0) | (bl > 0 ? 1 : 0);
      if (idx === 0 || idx === 15) continue;
      const top = () => [(i + tl / (tl - tr)) * cell, j * cell];
      const right = () => [(i + 1) * cell, (j + tr / (tr - br)) * cell];
      const bottom = () => [(i + bl / (bl - br)) * cell, (j + 1) * cell];
      const left = () => [i * cell, (j + tl / (tl - bl)) * cell];
      const segs = { 1: [[left, bottom]], 2: [[bottom, right]], 3: [[left, right]], 4: [[top, right]], 5: [[top, right], [left, bottom]], 6: [[top, bottom]], 7: [[top, left]], 8: [[top, left]], 9: [[top, bottom]], 10: [[top, left], [bottom, right]], 11: [[top, right]], 12: [[left, right]], 13: [[bottom, right]], 14: [[left, bottom]] }[idx];
      segs.forEach(([a, b]) => { const p = a(), q = b(); ctx2d.moveTo(p[0], p[1]); ctx2d.lineTo(q[0], q[1]); });
    }
    ctx2d.strokeStyle = c.ink; ctx2d.lineWidth = 1.6; ctx2d.setLineDash([6, 4]); ctx2d.stroke(); ctx2d.setLineDash([]);
  }

  function treeRects(tr) {
    const rects = new Array(tr.nNodes); rects[0] = [0, 0, 1, 1];
    const st = [0];
    while (st.length) {
      const nd = st.pop(); if (tr.feat[nd] < 0) continue;
      const [x0, y0, x1, y1] = rects[nd], t = tr.thr[nd], l = tr.left[nd], r = tr.right[nd];
      if (tr.feat[nd] === 0) { rects[l] = [x0, y0, t, y1]; rects[r] = [t, y0, x1, y1]; } else { rects[l] = [x0, y0, x1, t]; rects[r] = [x0, t, x1, y1]; }
      st.push(l, r);
    }
    return rects;
  }
  function getTree() {
    if (treeCache && treeCache.idx === state.treeIdx) return treeCache;
    const tr = model.f.trees[state.treeIdx];
    const mult = new Uint16Array(model.n);
    model.f.samples[state.treeIdx].forEach((i) => mult[i]++);
    const sel = state.selected, pt = [data.X[sel * 2], data.X[sel * 2 + 1]];
    treeCache = { idx: state.treeIdx, tr, rects: treeRects(tr), mult, path: model.f.pathNodes(state.treeIdx, pt), sel };
    return treeCache;
  }
  const PX = (x) => x * W, PY = (y) => (1 - y) * W;

  function drawTreeLayer(c) {
    const T = getTree(), tr = T.tr, L = state.depthShown;
    const k = Math.min(L, T.path.length - 1), nd = T.path[k], [x0, y0, x1, y1] = T.rects[nd];
    ctx2d.fillStyle = c.dark ? 'rgba(57,135,229,0.30)' : 'rgba(42,120,214,0.20)';
    ctx2d.fillRect(PX(x0), PY(y1), PX(x1) - PX(x0), PY(y0) - PY(y1));
    ctx2d.strokeStyle = c.accent; ctx2d.lineWidth = 2; ctx2d.strokeRect(PX(x0), PY(y1), PX(x1) - PX(x0), PY(y0) - PY(y1));
    for (let i = 0; i < tr.nNodes; i++) {
      if (tr.feat[i] < 0 || tr.depth[i] >= L) continue;
      const [a0, b0, a1, b1] = T.rects[i], t = tr.thr[i];
      ctx2d.beginPath();
      if (tr.feat[i] === 0) { ctx2d.moveTo(PX(t), PY(b0)); ctx2d.lineTo(PX(t), PY(b1)); } else { ctx2d.moveTo(PX(a0), PY(t)); ctx2d.lineTo(PX(a1), PY(t)); }
      ctx2d.strokeStyle = c.ink; ctx2d.globalAlpha = clamp(0.95 - tr.depth[i] * 0.09, 0.35, 0.95);
      ctx2d.lineWidth = Math.max(0.9, 2.6 - tr.depth[i] * 0.3); ctx2d.stroke();
    }
    ctx2d.globalAlpha = 1;
  }

  function drawPoints(c) {
    const { X, y, n } = data, flag = model.scores, thr = model.thr, treeMode = state.view === 'tree';
    const T = treeMode ? getTree() : null;
    const pass = (anom) => {
      for (let i = 0; i < n; i++) {
        if (!!y[i] !== anom) continue;
        const px = PX(X[i * 2]), py = PY(X[i * 2 + 1]);
        const anomColor = state.showLabels && y[i];
        ctx2d.globalAlpha = treeMode && !T.mult[i] ? 0.22 : 1;
        ctx2d.beginPath(); ctx2d.arc(px, py, anomColor ? 4.4 : 3.1, 0, 6.2832);
        ctx2d.fillStyle = anomColor ? c.s2 : c.ink2; ctx2d.fill();
        ctx2d.lineWidth = 1.2; ctx2d.strokeStyle = c.surface; ctx2d.stroke();
        if (treeMode && T.mult[i] > 1) { ctx2d.beginPath(); ctx2d.arc(px, py, 6.2, 0, 6.2832); ctx2d.strokeStyle = c.ink2; ctx2d.lineWidth = 1; ctx2d.stroke(); }
        if (!treeMode && flag[i] > thr) { ctx2d.beginPath(); ctx2d.arc(px, py, (anomColor ? 4.4 : 3.1) + 3, 0, 6.2832); ctx2d.strokeStyle = c.ink; ctx2d.lineWidth = 1.5; ctx2d.stroke(); }
      }
    };
    pass(false); pass(true);
    ctx2d.globalAlpha = 1;
    [state.hover, state.selected].forEach((idx, k) => {
      if (idx < 0 || idx >= n) return;
      const px = PX(X[idx * 2]), py = PY(X[idx * 2 + 1]);
      ctx2d.beginPath(); ctx2d.arc(px, py, k ? 10 : 8, 0, 6.2832);
      ctx2d.lineWidth = k ? 4 : 3; ctx2d.strokeStyle = c.surface; ctx2d.stroke();
      ctx2d.lineWidth = k ? 2 : 1.4; ctx2d.strokeStyle = k ? c.accent : c.ink; ctx2d.stroke();
    });
  }

  function drawMap() {
    if (!model) return;
    const c = ML.colors();
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx2d.clearRect(0, 0, W, W);
    ctx2d.fillStyle = c.surface; ctx2d.fillRect(0, 0, W, W);
    if (state.view === 'map') { if (state.showHeat) drawHeat(c); drawContour(c); } else drawTreeLayer(c);
    drawPoints(c);
    // marcos de referencia
    ctx2d.strokeStyle = c.grid; ctx2d.lineWidth = 1; ctx2d.setLineDash([2, 4]);
    ctx2d.strokeRect(0.5, 0.5, W - 1, W - 1); ctx2d.setLineDash([]);
    ctx2d.fillStyle = c.muted; ctx2d.font = '11px system-ui, sans-serif';
    ctx2d.fillText('x₁ →', W - 34, W - 8); ctx2d.save(); ctx2d.translate(12, 32); ctx2d.rotate(-Math.PI / 2); ctx2d.fillText('x₂ →', 0, 0); ctx2d.restore();
    const thrPct = clamp((model.thr - S_LO) / (S_HI - S_LO), 0, 1) * 100;
    $('#cbThr').style.left = thrPct + '%';
    $('#cbWrap').style.display = state.view === 'map' && state.showHeat ? '' : 'none';
    $('.legend').classList.toggle('no-labels', !state.showLabels);
  }

  function nearest(px, py, maxD) {
    const { X, n } = data; let best = -1, bd = maxD * maxD;
    for (let i = 0; i < n; i++) {
      const dx = PX(X[i * 2]) - px, dy = PY(X[i * 2 + 1]) - py, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  function wireCanvas() {
    const pos = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    canvas.addEventListener('pointermove', (e) => {
      if (!model || e.pointerType === 'touch') return;
      const [px, py] = pos(e), i = nearest(px, py, 14);
      if (i !== state.hover) { state.hover = i; drawMap(); }
      if (i >= 0) {
        const s = model.scores[i];
        ML.tip.show(`<span class="tip-h">Punto #${i + 1} · ${data.y[i] ? 'anómalo' : 'normal'} (real)</span>` +
          `<span class="tip-row"><span>Score s(x)</span><b>${fmt(s, 3)}</b></span><span class="tip-row"><span>Profundidad media E[h]</span><b>${fmt(ehOf(s), 2)}</b></span>` +
          `<span class="tip-row"><span>Decisión del modelo</span><b>${s > model.thr ? 'anómalo' : 'normal'}</b></span><span class="tip-sub">Clic para analizarlo (Anatomía y Un árbol)</span>`, e.clientX, e.clientY);
      } else if (state.addMode !== 'select') ML.tip.show(`Clic para agregar un punto ${state.addMode === 'anomaly' ? 'anómalo' : 'normal'}`, e.clientX, e.clientY);
      else ML.tip.hide();
    });
    canvas.addEventListener('pointerleave', () => { state.hover = -1; ML.tip.hide(); drawMap(); });
    canvas.addEventListener('pointerdown', (e) => {
      if (!model) return;
      const [px, py] = pos(e);
      if (state.addMode === 'select') {
        const i = nearest(px, py, e.pointerType === 'touch' ? 26 : 16);
        if (i >= 0) { state.selected = i; treeCache = null; renderSelection(); }
      } else {
        state.added.push({ x: clamp(px / W, 0.01, 0.99), y: clamp(1 - py / W, 0.01, 0.99), label: state.addMode === 'anomaly' ? 1 : 0 });
        scheduleRebuild();
      }
    });
  }

  /* =====================================================================
   *  Render: métricas, lectura, selección
   * ===================================================================== */
  const TILE_ORDER = { rank: ['auc', 'ap'], dec: ['precision', 'recall', 'f1', 'flagged'] };
  function buildTiles() {
    Object.entries(TILE_ORDER).forEach(([g, ids]) => {
      $(g === 'rank' ? '#tilesRank' : '#tilesDec').innerHTML = ids.map((id) => {
        const d = ML.IF_METRICS[id];
        return `<div class="tile" tabindex="0" data-m="${id}" data-tip="${esc(d.tip)}"><div class="t-label">${d.label}<i>?</i></div><div class="t-val">—</div><div class="delta flat"></div></div>`;
      }).join('');
    });
  }
  function deltaHTML(id, v, b) {
    if (!state.baseline || !Number.isFinite(v) || !Number.isFinite(b)) return ['delta flat', state.baseline ? '' : ''];
    if (id === 'flagged') { const d = v - b; return d === 0 ? ['delta flat', '= igual'] : ['delta flat', `${d > 0 ? '▲ +' : '▼ '}${d} vs ref.`]; }
    const d = v - b;
    if (Math.abs(d) < 0.0005) return ['delta flat', '= igual'];
    return [d > 0 ? 'delta up' : 'delta down', `${d > 0 ? '▲ +' : '▼ '}${d.toFixed(3)} vs ref.`];
  }
  function renderMetrics() {
    const m = metricsNow(), base = state.baseline ? state.baseline.m : null;
    $$('.tile').forEach((t) => {
      const id = t.dataset.m, v = m[id];
      t.querySelector('.t-val').innerHTML = id === 'flagged' ? `${v}<small> / ${m.n}</small>` : fmt(v, 3);
      const [cls, txt] = deltaHTML(id, v, base ? base[id] : NaN);
      const dl = t.querySelector('.delta'); dl.className = cls; dl.textContent = txt;
    });
    $('#baseBtn').textContent = state.baseline ? '📌 Actualizar referencia' : '📌 Fijar referencia';
    $('#baseClear').hidden = !state.baseline;
    $('#miniMetrics').innerHTML = [['ROC-AUC', fmt(m.auc, 3)], ['AP', fmt(m.ap, 3)], ['F1', fmt(m.f1, 3)], ['Marcados', `${m.flagged}/${m.n}`]]
      .map(([k, v]) => `<div class="mm"><span>${k}</span><b>${v}</b></div>`).join('');
    renderReading(m);
  }

  function tierAuc(a) { return a >= 0.98 ? 'casi perfecto' : a >= 0.93 ? 'muy bueno' : a >= 0.85 ? 'bueno' : a >= 0.7 ? 'flojo' : 'cercano al azar'; }
  function renderReading(m) {
    const el = $('#reading'), P = state.params, out = [];
    if (!model.cv) { el.innerHTML = '<p class="r-h">Lectura</p><p>Para calcular métricas hacen falta puntos de <b>ambas</b> clases. Agrega puntos normales y anómalos con el modo <i>+ Normal / + Anómalo</i>.</p>'; return; }
    const lc = state.lastChange;
    if (lc && lc.before) {
      const diffs = [];
      [['auc', 'ROC-AUC'], ['ap', 'AP'], ['f1', 'F1']].forEach(([k, name]) => {
        const d = m[k] - lc.before[k];
        if (Math.abs(d) >= 0.0005) diffs.push(`${name} ${fmt(lc.before[k], 3)} → <b>${fmt(m[k], 3)}</b> (${d > 0 ? '▲ +' : '▼ '}${d.toFixed(3)})`);
      });
      const nm = `<code>${PMAP[lc.id].code}</code>`;
      let s = `Cambiaste ${nm} de <b>${fmtParam(lc.id, lc.from)}</b> a <b>${fmtParam(lc.id, lc.to)}</b>: `;
      s += diffs.length ? diffs.join(' · ') + '.' : 'ninguna métrica cambió de forma apreciable.';
      if (lc.id === 'contamination' && Math.abs(m.auc - lc.before.auc) < 0.0005) s += ' ROC-AUC y AP no se mueven porque <b>contamination solo cambia el umbral</b>, no el ranking.';
      out.push(['Último cambio', s]);
    }
    const prev = model.pos / model.n;
    out.push(['Ranking', `ROC-AUC <b>${fmt(m.auc, 3)}</b>: ${tierAuc(m.auc)}. Precisión media (AP) <b>${fmt(m.ap, 3)}</b>; un modelo al azar tendría ≈ ${fmt(prev, 3)}.`]);
    const flagged = m.flagged, pos = model.pos;
    let t;
    if (P.contamination === 'auto') t = `Con <code>'auto'</code> el umbral es 0.5 y se marcan <b>${flagged}</b> puntos (${pct(flagged / m.n)}), pero las anomalías reales son ${pos} (${pct(prev)}). `;
    else t = `Con contamination = ${pct(P.contamination)} el umbral es ${fmt(model.thr, 3)} y se marcan <b>${flagged}</b> puntos; las anomalías reales son ${pos} (${pct(prev)}). `;
    if (flagged > pos * 1.3) t += `Se marcan de más: hay falsas alarmas (precisión ${fmt(m.precision, 2)}).`;
    else if (flagged < pos * 0.7) t += `Se marcan de menos: se escapan anomalías (recall ${fmt(m.recall, 2)}).`;
    else t += `El umbral está bien calibrado (precisión ${fmt(m.precision, 2)}, recall ${fmt(m.recall, 2)}).`;
    out.push(['Umbral', t]);
    const warn = [];
    if (model.f.nEstimators < 30) warn.push(`con solo <b>${model.f.nEstimators}</b> árboles el resultado es ruidoso: cambia la semilla y comprueba cuánto varía`);
    if (model.f.psi < 32) warn.push(`con ψ = <b>${model.f.psi}</b> cada árbol es demasiado tosco`);
    if (warn.length) out.push(['Cuidado', warn.join('; ') + '.']);
    el.innerHTML = out.map(([h, p]) => `<p><span class="r-h">${h}</span><br>${p}</p>`).join('');
  }

  function renderSelection() {
    const i = state.selected, s = model.scores[i], eh = ehOf(s);
    $('#selBox').innerHTML = `<h5>Punto seleccionado</h5>
      <div class="kv"><span>Punto</span><b>#${i + 1}</b></div>
      <div class="kv"><span>Real</span><b>${data.y[i] ? 'anómalo' : 'normal'}</b></div>
      <div class="kv"><span>Score s(x)</span><b>${fmt(s, 3)}</b></div>
      <div class="kv"><span>E[h(x)]</span><b>${fmt(eh, 2)}</b></div>
      <div class="kv"><span>Modelo</span><b>${s > model.thr ? 'marcado' : 'normal'}</b></div>`;
    renderTreeNote(); renderAnatomy(); renderConvergence(); drawMap();
  }
  function renderTreeNote() {
    const isTree = state.view === 'tree';
    $('#treeCtl').hidden = !isTree;
    if (!isTree) return;
    const f = model.f, T = getTree(), tr = T.tr, L = state.depthShown;
    $('#treeNum').textContent = `${state.treeIdx + 1} / ${f.nEstimators}`;
    const dr = $('#depthRange'); dr.max = f.maxDepth; if (+dr.value > f.maxDepth) { dr.value = f.maxDepth; state.depthShown = f.maxDepth; }
    $('#depthVal').textContent = state.depthShown;
    const k = Math.min(state.depthShown, T.path.length - 1), nd = T.path[k], leaf = k === T.path.length - 1 && tr.feat[nd] < 0;
    const h = f.pathLength(state.treeIdx, [data.X[state.selected * 2], data.X[state.selected * 2 + 1]], 0);
    let msg;
    if (leaf && tr.size[nd] === 1) msg = `Punto #${state.selected + 1}: <b>aislado en el nivel ${tr.depth[nd]}</b> de este árbol (h = ${fmt(h, 2)}). ${tr.depth[nd] <= 3 ? 'Se aisló rápido: comportamiento típico de una anomalía.' : 'Necesitó varios cortes: comportamiento típico de un punto normal.'}`;
    else if (leaf) msg = `Punto #${state.selected + 1}: llegó al límite de profundidad (${tr.depth[nd]}) con ${tr.size[nd]} puntos en su celda; se suma la corrección c(${tr.size[nd]}) = ${fmt(f.cTab[tr.size[nd]], 2)} → h = ${fmt(h, 2)}.`;
    else msg = `Nivel ${k}: la celda del punto seleccionado aún contiene <b>${tr.size[nd]}</b> de los ${f.psi} puntos de la muestra. Sigue subiendo los niveles para ver cuándo queda solo.`;
    msg += ` <span style="opacity:.8">Puntos atenuados = no entraron en la muestra de este árbol${f.bootstrap ? '; anillo = punto repetido (bootstrap)' : ''}.</span>`;
    $('#treeNote').innerHTML = msg;
  }

  /* =====================================================================
   *  Render: gráficas
   * ===================================================================== */
  function renderCharts() {
    if (!model) return;
    const { scores, thr, cv, conf } = model, y = data.y;
    Ch.scoreHist($('#histChart'), { scores, y, thr, hasBoth: !!cv });
    if (cv) {
      const op = { fpr: conf.fp / cv.neg, tpr: conf.recall, recall: conf.recall, precision: conf.tp + conf.fp ? conf.precision : 1 };
      Ch.curve($('#rocChart'), { kind: 'roc', c: cv, op });
      Ch.curve($('#prChart'), { kind: 'pr', c: cv, op, baseline: cv.pos / model.n });
      $('#rocTitle').textContent = `ROC · AUC = ${fmt(cv.auc, 3)}`;
      $('#prTitle').textContent = `Precisión–Recall · AP = ${fmt(cv.ap, 3)}`;
    } else {
      ['#rocChart', '#prChart'].forEach((s) => { $(s).innerHTML = '<div class="empty">Necesitas puntos de ambas clases.</div>'; });
    }
    renderAnatomy(); renderConvergence(); renderSweepChrome(); drawSweepFromCache();
  }

  function pickIndex(kind) {
    const o = order(model.scores);
    if (kind === 'top') return o[0];
    if (kind === 'median') return o[Math.floor(o.length / 2)];
    let best = 0, bd = Infinity;
    o.forEach((i) => { const d = Math.abs(model.scores[i] - model.thr); if (d < bd) { bd = d; best = i; } });
    return best;
  }
  function renderAnatomy() {
    const f = model.f, i = state.selected, x = [data.X[i * 2], data.X[i * 2 + 1]];
    const h = f.pathLengthsOf(x);
    let mean = 0; h.forEach((v) => { mean += v; }); mean /= h.length;
    const s = f.scoreFromMeanPath(mean);
    Ch.pathHist($('#anatChart'), { h, mean, cPsi: f.cPsi });
    const verdict = s > model.thr ? `supera el umbral ${fmt(model.thr, 3)} → <b>marcado como anómalo</b>` : `no supera el umbral ${fmt(model.thr, 3)} → <b>normal</b>`;
    $('#formulaBox').innerHTML = `<div>Punto <b>#${i + 1}</b> (real: ${data.y[i] ? 'anómalo' : 'normal'})</div>
      <div class="fx">E[h(x)] = ${fmt(mean, 2)} <span style="opacity:.7">(promedio de ${f.nEstimators} árboles)</span></div>
      <div class="fx">c(ψ) = c(${f.psi}) = ${fmt(f.cPsi, 2)}</div>
      <div class="fx">s(x) = 2^(−${fmt(mean, 2)} / ${fmt(f.cPsi, 2)}) = <b>${fmt(s, 3)}</b></div>
      <div>${s > 0.5 ? 'Se aísla antes de lo esperable' : 'Se aísla después de lo esperable'} (E[h] ${mean < f.cPsi ? '<' : '≥'} c(ψ)) y ${verdict}.</div>`;
  }
  function renderConvergence() {
    const f = model.f, hOf = (i) => f.pathLengthsOf([data.X[i * 2], data.X[i * 2 + 1]]);
    const c = ML.colors();
    const top = pickIndex('top'), bor = pickIndex('border'), med = pickIndex('median');
    const series = [
      { name: 'más anómalo', h: hOf(top), color: c.s2 },
      { name: 'cerca del umbral', h: hOf(bor), color: c.s3 },
      { name: 'típico', h: hOf(med), color: c.s1 },
    ];
    if (![top, bor, med].includes(state.selected)) series.push({ name: `seleccionado #${state.selected + 1}`, h: hOf(state.selected), color: c.ink, dash: true });
    Ch.convergence($('#convChart'), { series, thr: model.thr, T: f.nEstimators, cPsi: f.cPsi });
  }

  /* ---------- Barrido ---------- */
  const scoreCache = new Map();
  let sweepToken = 0, sweepState = null;
  function sweepValues(id) {
    const spec = SWEEPS[id];
    let vals = spec.values.slice();
    if (id === 'maxSamples') vals = vals.filter((v) => v <= data.n);
    const cur = state.params[id];
    if ((spec.kind === 'lin' || spec.kind === 'log') && typeof cur === 'number' && !vals.includes(cur)) { vals.push(cur); vals.sort((a, b) => a - b); }
    return vals;
  }
  function scoresFor(p) {
    const key = [dataVersion, p.nEstimators, p.maxSamples, p.maxFeatures, p.bootstrap ? 1 : 0, p.seed].join('|');
    let s = scoreCache.get(key);
    if (!s) {
      s = IF.fit(data.X, data.n, 2, p).scores(data.X, data.n);
      if (scoreCache.size > 500) scoreCache.clear();
      scoreCache.set(key, s);
    }
    return s;
  }
  async function runSweep() {
    if (!data) return;
    const token = ++sweepToken, id = state.active, spec = SWEEPS[id], vals = sweepValues(id);
    const results = [];
    sweepState = { id, spec, results, total: vals.length, running: true, current: state.params[id], token };
    drawSweep();
    for (const v of vals) {
      const metrics = { auc: [], ap: [], f1: [] };
      for (let r = 0; r < spec.reps; r++) {
        const p = { ...state.params };
        if (id === 'seed') p.seed = v; else { p[id] = v; p.seed = state.params.seed + 1000 * r; }
        const s = scoresFor(p), cv = M.curves(s, data.y);
        if (!cv) continue;
        const thr = IF.threshold(s, p.contamination);
        metrics.auc.push(cv.auc); metrics.ap.push(cv.ap); metrics.f1.push(M.confusion(s, data.y, thr).f1);
      }
      if (token !== sweepToken) return;
      if (metrics.auc.length) results.push({ x: v, metrics });
      drawSweep();
      await new Promise((res) => setTimeout(res, 0));
      if (token !== sweepToken) return;
    }
    sweepState.running = false;
    drawSweep();
  }
  function drawSweepFromCache() { if (sweepState) drawSweep(); }
  function drawSweep() {
    const s = sweepState; if (!s) return;
    const mLabel = { auc: 'ROC-AUC', ap: 'Precisión media (AP)', f1: 'F1' }[state.sweepMetric];
    Ch.sweep($('#sweepChart'), { spec: s.spec, results: s.results, metric: state.sweepMetric, metricLabel: mLabel, current: s.current, running: s.running, total: s.total,
      mark: s.id === 'contamination' && model ? { x: model.pos / model.n, label: `% real (${(100 * model.pos / model.n).toFixed(1)}%)` } : null });
    // tabla accesible
    const rows = s.results.map((r) => {
      const a = r.metrics[state.sweepMetric], m = a.reduce((x, v) => x + v, 0) / a.length, sd = Math.sqrt(a.reduce((x, v) => x + (v - m) ** 2, 0) / a.length);
      return `<tr><td>${esc(String(s.spec.tickLabel ? s.spec.tickLabel(r.x) : r.x))}</td><td>${fmt(m, 3)}</td><td>${fmt(sd, 3)}</td><td>${fmt(Math.min(...a), 3)}</td><td>${fmt(Math.max(...a), 3)}</td></tr>`;
    }).join('');
    $('#sweepTable').innerHTML = `<table><thead><tr><th>${s.spec.code}</th><th>media</th><th>desv.</th><th>mín</th><th>máx</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  function renderSweepChrome() {
    const spec = SWEEPS[state.active];
    $('#sweepTitle').innerHTML = `Barrido de <code>${spec.code}</code>`;
    $('#sweepHint').textContent = SWEEP_HINT[state.active];
    $$('#sweepMetricSeg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.m === state.sweepMetric)));
  }

  /* =====================================================================
   *  Escenarios guiados
   * ===================================================================== */
  function buildLessons() {
    const host = $('#lessonChips');
    host.innerHTML = ML.IF_LESSONS.map((l) => `<button type="button" role="tab" data-id="${l.id}" aria-selected="false">${l.title}<small>${l.tag}</small></button>`).join('');
    $$('button', host).forEach((b) => b.addEventListener('click', () => applyLesson(b.dataset.id)));
    $('#lessonBody').innerHTML = '<p class="lesson-empty">Elige un escenario para cargar los datos y parámetros adecuados y ver qué observar.</p>';
  }
  function applyLesson(id) {
    const l = ML.IF_LESSONS.find((x) => x.id === id), s = l.setup;
    state.lesson = id; state.dataset = s.dataset; state.added = []; state.selected = -1; state.dataSeed = 11; state.anomalyFrac = 0.06; state.nPoints = 500;
    state.params = { ...DEFAULTS, ...s.params };
    if (typeof s.params.contamination === 'number') state.lastContam = s.params.contamination;
    state.lastChange = null; state.baseline = null;
    state.view = s.view || 'map'; syncViewButtons();
    state.active = s.active; state.open = s.active;
    state.sweepMetric = s.sweepMetric || DEFAULT_SWEEP_METRIC[s.active] || 'ap';
    $$('#lessonChips button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.id === id)));
    $('#lessonBody').innerHTML = `<div class="lb-title">${l.title}</div>
      <div><h4>Qué hacer</h4><ol>${l.steps.map((x) => `<li>${x}</li>`).join('')}</ol></div>
      <div><h4>Qué observar</h4><p>${l.look}</p></div>
      <div><button class="btn" type="button" id="lessonReset">↺ Reiniciar escenario</button></div>`;
    $('#lessonReset').addEventListener('click', () => applyLesson(id));
    scheduleRebuild();
    renderSweepChrome();
    $('#lab').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* =====================================================================
   *  Vistas, toggles y hash
   * ===================================================================== */
  function syncViewButtons() {
    $('#viewMap').setAttribute('aria-pressed', String(state.view === 'map'));
    $('#viewTree').setAttribute('aria-pressed', String(state.view === 'tree'));
  }
  function scrollToCard(id) { const el = document.getElementById(id); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }

  function wireUI() {
    $('#viewMap').addEventListener('click', () => { state.view = 'map'; syncViewButtons(); renderTreeNote(); drawMap(); });
    $('#viewTree').addEventListener('click', () => { state.view = 'tree'; syncViewButtons(); treeCache = null; renderTreeNote(); drawMap(); });
    $('#togHeat').addEventListener('click', (e) => { state.showHeat = !state.showHeat; e.currentTarget.setAttribute('aria-pressed', String(state.showHeat)); drawMap(); });
    $('#togLabels').addEventListener('click', (e) => { state.showLabels = !state.showLabels; e.currentTarget.setAttribute('aria-pressed', String(state.showLabels)); drawMap(); });
    $$('[data-mode]').forEach((b) => b.addEventListener('click', () => {
      state.addMode = b.dataset.mode; $$('[data-mode]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      canvas.style.cursor = state.addMode === 'select' ? 'crosshair' : 'copy';
    }));
    const stepTree = (d) => { state.treeIdx = (state.treeIdx + d + model.f.nEstimators) % model.f.nEstimators; treeCache = null; renderTreeNote(); drawMap(); };
    $('#treePrev').addEventListener('click', () => stepTree(-1));
    $('#treeNext').addEventListener('click', () => stepTree(1));
    $('#depthRange').addEventListener('input', (e) => { state.depthShown = +e.target.value; renderTreeNote(); drawMap(); });
    $('#baseBtn').addEventListener('click', () => { state.baseline = { m: metricsNow() }; renderMetrics(); });
    $('#baseClear').addEventListener('click', () => { state.baseline = null; renderMetrics(); });
    $$('#sweepMetricSeg button').forEach((b) => b.addEventListener('click', () => { state.sweepMetric = b.dataset.m; renderSweepChrome(); drawSweep(); }));
    $$('[data-pick]').forEach((b) => b.addEventListener('click', () => { state.selected = pickIndex(b.dataset.pick); treeCache = null; renderSelection(); }));
    window.addEventListener('themechange', () => { drawMap(); renderConvergence(); });
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (resizeCanvas() && model) { drawMap(); renderCharts(); } }, 120); });
    if ('ResizeObserver' in window) new ResizeObserver(ML.debounce(() => { if (resizeCanvas() && model) { drawMap(); renderCharts(); } }, 100)).observe($('.canvas-wrap'));
  }

  function writeHash() {
    const P = state.params;
    const h = new URLSearchParams({ ds: state.dataset, np: state.nPoints, fr: (state.anomalyFrac * 100).toFixed(1), dsd: state.dataSeed, t: P.nEstimators, psi: P.maxSamples, c: P.contamination === 'auto' ? 'auto' : (P.contamination * 100).toFixed(1), mf: P.maxFeatures, bs: P.bootstrap ? 1 : 0, seed: P.seed, p: state.active });
    try { history.replaceState(null, '', '#' + h.toString()); } catch (e) { /* file:// u otros entornos */ }
  }
  function readHash() {
    if (!location.hash || location.hash.length < 4) return;
    const h = new URLSearchParams(location.hash.slice(1));
    if (!h.has('ds')) return;
    if (D.DEFS[h.get('ds')]) state.dataset = h.get('ds');
    const num = (k, d) => (h.has(k) && Number.isFinite(+h.get(k)) ? +h.get(k) : d);
    state.nPoints = [200, 500, 1000].includes(num('np', 500)) ? num('np', 500) : 500;
    state.anomalyFrac = clamp(num('fr', 6), 1, 15) / 100; state.dataSeed = num('dsd', 11);
    const P = state.params;
    P.nEstimators = clamp(Math.round(num('t', 100)), 1, 500); P.maxSamples = clamp(Math.round(num('psi', 256)), 2, 5000);
    P.contamination = h.get('c') === 'auto' || !h.has('c') ? 'auto' : clamp(num('c', 10), 0.5, 50) / 100;
    if (P.contamination !== 'auto') state.lastContam = P.contamination;
    P.maxFeatures = num('mf', 1) < 1 ? 0.5 : 1; P.bootstrap = h.get('bs') === '1'; P.seed = Math.max(0, Math.floor(num('seed', 42)));
    if (PMAP[h.get('p')]) { state.active = h.get('p'); state.open = state.active; }
    state.sweepMetric = DEFAULT_SWEEP_METRIC[state.active] || 'ap';
  }

  /* =====================================================================
   *  Render global + init
   * ===================================================================== */
  function renderAll() {
    syncDataControls(); syncParamCards(); renderMetrics();
    renderSelection();     // incluye anatomía, convergencia, nota de árbol y mapa
    renderCharts();
  }

  function init() {
    ML.initTheme();
    readHash();
    buildTiles(); buildParamCards(); buildDataControls(); buildLessons(); wireUI(); wireCanvas();
    resizeCanvas();
    buildData(); compute(); renderAll(); runSweep(); writeHash();
    // Se expone para depuración/pruebas en clase
    ML.lab = { state, get model() { return model; }, get data() { return data; } };
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
