/* Controlador del laboratorio de Gradient Boosting: estado, entrenamiento por rondas y renderizado. */
(function () {
  'use strict';
  const ML = window.MLLab;
  const GBM = ML.GBM, D = ML.gbData, M = ML.metrics, Ch = ML.charts;
  const { $, $$, clamp, fmt, esc } = ML;
  const PARAMS = ML.GB_PARAMS, PMAP = Object.fromEntries(PARAMS.map((p) => [p.id, p]));

  const DEF = {
    dataset: 'blobs', n: 300, noise: 0, minor: 0.5,
    nEstimators: 40, lr: 0.3, maxDepth: 3, minChildWeight: 1, gamma: 0, lambda: 1, subsample: 1, seed: 1,
  };
  const STOPS = {
    nEstimators: [0, 1, 2, 3, 5, 10, 20, 30, 40, 60, 80, 120, 160, 200, 300],
    lr: [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1],
    maxDepth: [1, 2, 3, 4, 5, 6, 8],
    minChildWeight: [0.25, 0.5, 1, 2, 5, 10, 20, 40],
    gamma: [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
    lambda: [0, 0.1, 0.5, 1, 2, 5, 10, 20],
    subsample: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
  };
  const SWEEPS = {
    nEstimators: { code: 'n_estimators', kind: 'log', values: [1, 3, 5, 10, 20, 40, 80, 160, 300], axisLabel: 'n_estimators (escala log)', reps: 2 },
    lr: { code: 'learning_rate', kind: 'log', values: [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9], tickLabel: (v) => String(v), axisLabel: 'learning_rate (escala log)', reps: 2 },
    maxDepth: { code: 'max_depth', kind: 'lin', values: [1, 2, 3, 4, 5, 6, 8], axisLabel: 'max_depth', reps: 2 },
    minChildWeight: { code: 'min_child_weight', kind: 'log', values: [0.25, 0.5, 1, 2, 5, 10, 20, 40], axisLabel: 'min_child_weight (escala log)', reps: 2 },
    gamma: { code: 'gamma', kind: 'lin', values: [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5], axisLabel: 'gamma', reps: 2 },
    lambda: { code: 'lambda', kind: 'lin', values: [0, 0.1, 0.5, 1, 2, 5, 10, 20], axisLabel: 'lambda (regularización L2)', reps: 2 },
    subsample: { code: 'subsample', kind: 'lin', values: [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1], axisLabel: 'subsample', reps: 3 },
    seed: { code: 'random_state', kind: 'cat', values: [1, 2, 3, 4, 5, 6, 7, 8], band: true, axisLabel: 'random_state (cada punto = una semilla)', reps: 1 },
  };
  const SWEEP_HINT = {
    nEstimators: 'Cada valor entrena desde cero. Fíjate en dónde la pérdida de validación deja de bajar y empieza a subir.',
    lr: 'La pareja clásica: con η bajo hacen falta más rondas; con η alto, menos (pero sobreajusta antes).',
    maxDepth: 'Más profundidad reduce el sesgo hasta cierto punto; pasado ese punto, solo añade sobreajuste.',
    minChildWeight: 'Solo se nota cuando hay hojas pequeñas: profundidad alta y/o pocos datos.',
    gamma: 'Poda: a partir de cierto γ ningún corte se acepta y el modelo deja de aprender.',
    lambda: 'Encoge las hojas. Su efecto es más suave que el de gamma, pero en la misma dirección.',
    subsample: 'Menos filas por ronda = más varianza entre semillas, a veces mejor generalización.',
    seed: 'Cada punto es el mismo modelo con otra semilla: su dispersión es la varianza atribuible al azar.',
  };
  const SWEEP_METRICS = { val: 'Logloss (validación)', train: 'Logloss (entrenamiento)', gap: 'Brecha train–val', auc: 'ROC-AUC (val)', ap: 'AP (val)', acc: 'Exactitud (val)' };
  const LOWER_IS_BETTER = { val: 1, train: 1, gap: 1 };
  const GRID_N = 56;

  const state = { ...DEF, active: 'nEstimators', open: 'nEstimators', sweepMetric: 'val', curveMetric: 'logloss', showHeat: true, showPoints: true, treeIdx: 0, sel: -1, lesson: null };
  let data = null, model = null, valP = null, trainP = null, gridP = null, training = false, trainToken = 0;

  /* =====================================================================
   *  Datos y modelo
   * ===================================================================== */
  function buildData() {
    data = D.generate(state.dataset, state.n, state.seed + 7919, state.noise, state.minor);
    state.sel = -1;
  }
  function modelOpts(over) {
    const s = Object.assign({}, state, over || {});
    return { nEstimators: s.nEstimators, learningRate: s.lr, maxDepth: s.maxDepth, minChildWeight: s.minChildWeight, gamma: s.gamma, lambda: s.lambda, subsample: s.subsample, seed: s.seed };
  }
  /** Reajusta el modelo completo: en boosting los primeros T árboles no cambian al pedir más rondas. */
  function refit() {
    const t = data.train, v = data.val;
    model = GBM.fit(t.X, t.y, t.n, 2, modelOpts(), [{ X: v.X, y: v.y, n: v.n }]);
    trainP = model.predictProba(t.X, t.n);
    valP = model.predictProba(v.X, v.n);
    gridP = predictGrid();
    state.treeIdx = clamp(state.treeIdx, 0, Math.max(0, model.trees.length - 1));
  }
  function predictGrid() {
    const G = GRID_N, XY = new Float64Array(G * G * 2);
    for (let r = 0; r < G; r++) for (let k = 0; k < G; k++) {
      XY[(r * G + k) * 2] = (k + 0.5) / G;
      XY[(r * G + k) * 2 + 1] = 1 - (r + 0.5) / G;
    }
    return model.predictProba(XY, G * G);
  }
  /** Añade k rondas de boosting sin bloquear la interfaz. */
  async function train(k) {
    if (training) return;
    training = true;
    const token = ++trainToken;
    setTrainingUI(true);
    const target = Math.min(300, state.nEstimators + k);
    const chunk = Math.max(1, Math.round(k / 6));
    while (state.nEstimators < target) {
      state.nEstimators = Math.min(target, state.nEstimators + chunk);
      refit(); renderAll();
      await new Promise((r) => requestAnimationFrame(r));
      if (token !== trainToken) { training = false; return; }
    }
    training = false; setTrainingUI(false);
    renderAll(); scheduleSweep();
  }
  function setTrainingUI(on) {
    $$('.train-btn').forEach((b) => { b.disabled = on; });
    const el = $('#trainState'); if (el) el.textContent = on ? 'entrenando…' : '';
  }
  function reset() {
    trainToken++; training = false; setTrainingUI(false);
    state.nEstimators = 0; refit(); renderAll(); scheduleSweep();
  }

  /* =====================================================================
   *  Métricas
   * ===================================================================== */
  function metricsNow() {
    const h = model.hist.length ? model.hist[model.hist.length - 1] : null;
    const v = data.val;
    const cv = M.curves(valP, v.y);
    return {
      trainLoss: h ? h.train : GBM.logloss(new Float64Array(data.train.n).fill(model.base), data.train.y, data.train.n),
      valLoss: h ? h.val0 : GBM.logloss(new Float64Array(v.n).fill(model.base), v.y, v.n),
      valAcc: h ? h.val0Acc : GBM.accuracy(new Float64Array(v.n).fill(model.base), v.y, v.n),
      trainAcc: h ? h.trainAcc : NaN,
      auc: cv ? cv.auc : NaN, ap: cv ? cv.ap : NaN,
      get gap() { return this.valLoss - this.trainLoss; },
    };
  }
  /** Ronda con la menor pérdida de validación (lo que elegiría early stopping). */
  function bestRound() {
    if (!model.hist.length) return 0;
    let bi = 0;
    model.hist.forEach((h, i) => { if (h.val0 < model.hist[bi].val0) bi = i; });
    return model.hist[bi].round;
  }

  /* =====================================================================
   *  Controles
   * ===================================================================== */
  function fmtParam(id, v = state[id]) {
    if (id === 'nEstimators') return String(v);
    if (id === 'subsample') return v === 1 ? 'todas' : String(v);
    if (id === 'maxDepth') return v === 1 ? '1 (tocón)' : String(v);
    return String(v);
  }
  function setParam(id, v) {
    if (state[id] === v) return;
    state[id] = v;
    openParam(id, false);
    trainToken++; training = false; setTrainingUI(false);
    if (id === 'seed') buildData();
    refit(); renderAll(); scheduleSweep();
  }
  function openParam(id, toggle) {
    const changed = state.active !== id;
    state.active = id;
    state.open = toggle && state.open === id ? null : id;
    $$('.pcard').forEach((el) => el.classList.toggle('active', el.dataset.id === state.open));
    if (changed) { renderSweepChrome(); scheduleSweep(); }
  }
  function controlHTML(p) {
    if (p.id === 'nEstimators') {
      return `<div class="ep-row">
        <button class="btn sm primary train-btn" type="button" data-ep="1">+1 ronda</button>
        <button class="btn sm train-btn" type="button" data-ep="10">+10</button>
        <button class="btn sm train-btn" type="button" data-ep="50">+50</button>
        <button class="btn sm" type="button" data-reset>↺ Reiniciar</button>
        <span class="train-state" id="trainState"></span></div>
        <input type="range" min="0" max="${STOPS.nEstimators.length - 1}" step="1" aria-label="n_estimators">
        <div class="p-ticks"><span>0</span><span>300</span></div>`;
    }
    if (p.id === 'seed') return `<div class="seed-row"><input type="number" min="0" max="9999" step="1" aria-label="random_state"><button class="btn sm" type="button" data-dice>🎲 Otra semilla</button></div>`;
    const s = STOPS[p.id];
    return `<input type="range" min="0" max="${s.length - 1}" step="1" aria-label="${esc(p.code)}"><div class="p-ticks"><span>${fmtParam(p.id, s[0])}</span><span>${fmtParam(p.id, s[s.length - 1])}</span></div>`;
  }
  function buildParams() {
    const host = $('#paramList');
    host.innerHTML = '';
    PARAMS.forEach((p) => {
      const el = document.createElement('div');
      el.className = 'pcard'; el.dataset.id = p.id;
      const tip = `<span class="tip-h">${esc(p.title)} · <code>${esc(p.code)}</code></span>${p.short}<span class="tip-sub">Clic para la explicación completa</span>`;
      el.innerHTML = `<button class="p-head" type="button" aria-expanded="false" data-tip="${esc(tip)}">
          <span class="p-sym" aria-hidden="true">${p.symbol}</span>
          <span class="p-title"><b>${p.title}</b><code>${esc(p.code)}</code></span>
          <span class="p-val" data-val></span><span class="p-chev" aria-hidden="true">▾</span></button>
        <div class="p-ctl" data-tip="${esc(tip)}">${controlHTML(p)}</div>
        <div class="p-body"><p class="p-short">${p.short}</p><dl>
          <dt>¿Qué es?</dt><dd>${p.what}</dd><dt>¿Qué controla?</dt><dd>${p.controls}</dd>
          <dt>${p.upLabel}</dt><dd class="up">${p.up}</dd><dt>${p.downLabel}</dt><dd class="down">${p.down}</dd>
          <dt>Valor típico</dt><dd>${p.typical}</dd><dt>En la práctica</dt><dd>${p.real}</dd><dt>Dónde se nota</dt><dd>${p.watch}</dd></dl>
          <div class="p-live" data-live></div><button class="btn sm p-go" type="button" data-go>Ver dónde se nota ↓</button></div>`;
      host.appendChild(el);
      el.querySelector('.p-head').addEventListener('click', () => openParam(p.id, true));
      el.querySelector('[data-go]').addEventListener('click', () => { const t = document.getElementById(p.target); if (t) t.scrollIntoView({ behavior: 'smooth', block: 'center' }); });
      if (p.id === 'nEstimators') {
        $$('[data-ep]', el).forEach((b) => b.addEventListener('click', () => train(+b.dataset.ep)));
        el.querySelector('[data-reset]').addEventListener('click', reset);
        el.querySelector('input[type=range]').addEventListener('input', (e) => setParam('nEstimators', STOPS.nEstimators[+e.target.value]));
      } else if (p.id === 'seed') {
        const inp = el.querySelector('input');
        inp.addEventListener('change', () => setParam('seed', Math.max(0, Math.floor(+inp.value || 0))));
        el.querySelector('[data-dice]').addEventListener('click', () => { const v = Math.floor(Math.random() * 999); inp.value = v; setParam('seed', v); });
      } else {
        el.querySelector('input[type=range]').addEventListener('input', (e) => setParam(p.id, STOPS[p.id][+e.target.value]));
      }
    });
  }
  function syncParams() {
    PARAMS.forEach((p) => {
      const el = $(`.pcard[data-id="${p.id}"]`);
      el.querySelector('[data-val]').textContent = fmtParam(p.id);
      const rng = el.querySelector('input[type=range]');
      if (rng && STOPS[p.id]) {
        let i = STOPS[p.id].indexOf(state[p.id]);
        if (i < 0) { let bd = Infinity; STOPS[p.id].forEach((v, k) => { const d2 = Math.abs(v - state[p.id]); if (d2 < bd) { bd = d2; i = k; } }); }
        rng.value = i >= 0 ? i : 0;
      }
      if (p.id === 'seed') { const inp = el.querySelector('input'); if (document.activeElement !== inp) inp.value = state.seed; }
      el.querySelector('[data-live]').innerHTML = p.live(state, model);
    });
  }

  function buildDataControls() {
    const list = $('#dsList');
    const defs = Object.entries(D.SHAPES).sort((a, b) => a[1].order - b[1].order);
    list.innerHTML = defs.map(([id, d]) => `<button type="button" role="radio" data-id="${id}">${esc(d.name)}</button>`).join('');
    $$('button', list).forEach((b) => b.addEventListener('click', () => { state.dataset = b.dataset.id; buildData(); refit(); renderAll(); scheduleSweep(); }));
    $('#nSel').addEventListener('change', (e) => { state.n = +e.target.value; buildData(); refit(); renderAll(); scheduleSweep(); });
    $('#minorSel').addEventListener('change', (e) => { state.minor = +e.target.value; buildData(); refit(); renderAll(); scheduleSweep(); });
    $('#noiseRange').addEventListener('input', (e) => {
      state.noise = +e.target.value / 100; $('#noiseVal').textContent = (+e.target.value) + '%';
      buildData(); refit(); renderAll(); scheduleSweep();
    });
  }
  function syncData() {
    $$('#dsList button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.id === state.dataset)));
    const pos = data.nPos, n = data.train.n;
    $('#dsBlurb').innerHTML = D.SHAPES[state.dataset].blurb +
      ` <b>${n} puntos de entrenamiento</b> (${(100 * pos / n).toFixed(0)}% de clase 1) y <b>${data.val.n} de validación</b>.` +
      (state.noise > 0 ? ` <span class="an">${Math.round(state.noise * 100)}% de las etiquetas están cambiadas al azar.</span>` : '');
    $('#nSel').value = String(state.n); $('#minorSel').value = String(state.minor);
    $('#noiseRange').value = Math.round(state.noise * 100); $('#noiseVal').textContent = Math.round(state.noise * 100) + '%';
  }

  /* =====================================================================
   *  Render
   * ===================================================================== */
  function renderTiles() {
    const m = metricsNow();
    const groups = { fit: ['trainLoss', 'valLoss', 'gap'], rank: ['auc', 'ap'], dec: ['valAcc'] };
    Object.entries(groups).forEach(([g, ids]) => {
      const host = $('#tiles-' + g); if (!host) return;
      host.innerHTML = ids.map((id) => {
        const d = ML.GB_METRICS[id], v = m[id];
        return `<div class="tile" tabindex="0" data-tip="${esc(d.tip)}"><div class="t-label">${d.label}<i>?</i></div><div class="t-val">${Number.isFinite(v) ? fmt(v, 3) : '—'}</div></div>`;
      }).join('');
    });
    $('#miniMetrics').innerHTML = [['Rondas', state.nEstimators], ['Logloss val', fmt(m.valLoss, 3)], ['AUC val', fmt(m.auc, 3)], ['Brecha', fmt(m.gap, 3)]]
      .map(([k, v]) => `<div class="mm"><span>${k}</span><b>${v}</b></div>`).join('');
  }

  function renderMap() {
    Ch.decisionMap($('#mapCanvas'), {
      grid: gridP, G: GRID_N, X: data.train.X, y: data.train.y, yTrue: data.train.yTrue,
      n: data.train.n, pred: trainP, showHeat: state.showHeat, showPoints: state.showPoints, sel: state.sel,
    });
    $('#togHeat').setAttribute('aria-pressed', String(state.showHeat));
    $('#togPoints').setAttribute('aria-pressed', String(state.showPoints));
  }

  function renderCurve() {
    Ch.lossCurve($('#curveChart'), { hist: model.hist, running: training, bestRound: bestRound(), metric: state.curveMetric });
    $$('#curveMetricSeg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.m === state.curveMetric)));
  }

  function renderImportance() {
    Ch.importance($('#impChart'), { gain: model.gainImportance, names: ['x₁', 'x₂'] });
  }

  function renderEval() {
    const v = data.val;
    Ch.scoreHist($('#histChart'), { scores: valP, y: v.y, thr: 0.5, hasBoth: true });
    const cv = M.curves(valP, v.y);
    if (cv) {
      const op = M.confusion(valP, v.y, 0.5);
      const opPoint = { fpr: op.fp / (op.fp + op.tn || 1), tpr: op.recall, recall: op.recall, precision: op.precision };
      Ch.curve($('#rocChart'), { kind: 'roc', c: cv, op: opPoint });
      Ch.curve($('#prChart'), { kind: 'pr', c: cv, op: opPoint, baseline: cv.pos / (cv.pos + cv.neg) });
    }
    Ch.calibration($('#calChart'), { p: valP, y: v.y, n: v.n });
  }

  function renderTree() {
    const T = model.trees.length;
    const host = $('#card-tree');
    if (!T) { $('#treeChart').innerHTML = '<div class="empty">Entrena al menos una ronda para ver el árbol.</div>'; $('#treeNum').textContent = '0 / 0'; $('#treeNote').textContent = ''; return; }
    const i = clamp(state.treeIdx, 0, T - 1);
    const tr = model.trees[i];
    Ch.treeSplits($('#treeChart'), { tree: tr, round: i + 1 });
    $('#treeNum').textContent = `${i + 1} / ${T}`;
    const leaves = Array.from(tr.weight).filter((_, k) => tr.feat[k] < 0);
    const mx = Math.max(...leaves.map(Math.abs));
    $('#treeNote').innerHTML = `Este árbol tiene <b>${leaves.length}</b> hoja(s); su corrección más fuerte es <b>${fmt(mx, 2)}</b>, que se aplica multiplicada por η = ${state.lr} → <b>${fmt(mx * state.lr, 3)}</b> en la predicción acumulada. ` +
      (leaves.length === 1 ? 'Con una sola hoja, el árbol no encontró ningún corte que superara <code>gamma</code>: esta ronda no aporta nada.' : 'Naranja = empuja hacia la clase 1; azul = hacia la clase 0.');
    host.hidden = false;
  }

  function renderRead() {
    const m = metricsNow(), out = [];
    if (state.nEstimators === 0) {
      $('#reading').innerHTML = '<p><span class="r-h">Sin entrenar</span><br>Con 0 rondas el modelo solo predice la proporción base de cada clase: el mapa es de un solo color. Pulsa <b>+1 ronda</b> para ver aparecer la primera frontera.</p>';
      return;
    }
    const br = bestRound(), last = model.hist[model.hist.length - 1];
    out.push(['Ajuste', `Tras <b>${state.nEstimators}</b> rondas: logloss de entrenamiento <b>${fmt(m.trainLoss, 3)}</b> y de validación <b>${fmt(m.valLoss, 3)}</b> (exactitud ${fmt(m.valAcc, 3)}, ROC-AUC ${fmt(m.auc, 3)}).`]);
    if (m.gap > 0.12 && br < state.nEstimators * 0.75) {
      out.push(['⚠ Sobreajuste', `La pérdida de validación tocó su mínimo en la <b>ronda ${br}</b> y desde entonces empeora, mientras la de entrenamiento sigue bajando: el modelo está memorizando. <b>Early stopping</b> habría parado en ${br}. Otras salidas: bajar <code>max_depth</code> (ahora ${state.maxDepth}), bajar <code>learning_rate</code> (ahora ${state.lr}) o subir <code>gamma</code>/<code>lambda</code>.`]);
    } else if (m.gap < 0.02 && last.train > 0.35) {
      out.push(['Subajuste', `Entrenamiento y validación van casi igual y ambos siguen altos: al modelo le falta capacidad o rondas. Sube <code>n_estimators</code>, sube <code>max_depth</code> o sube <code>learning_rate</code>.`]);
    } else if (br === state.nEstimators) {
      out.push(['¿Sigue mejorando?', `El mejor valor de validación es <b>la última ronda</b>: todavía no has llegado al punto de sobreajuste. Añadir más rondas probablemente siga ayudando.`]);
    } else {
      out.push(['Punto de parada', `El mínimo de validación está en la ronda <b>${br}</b> de ${state.nEstimators}. La diferencia con la última ronda es pequeña (${fmt(m.valLoss - model.hist[br - 1].val0, 3)}), así que estás cerca del punto óptimo.`]);
    }
    if (state.maxDepth === 1 && state.dataset === 'xor') {
      out.push(['Profundidad 1 en el tablero', `Con <span data-term="tocon">tocones</span>, cada árbol corta por una sola variable, y en el tablero ninguna variable por separado informa: el modelo <b>no puede</b> mejorar por muchas rondas que sumes. Sube <code>max_depth</code> a 2.`]);
    }
    if (state.minor < 0.3) {
      out.push(['Clase rara', `Solo el ${(100 * data.nPos / data.train.n).toFixed(0)}% de los datos son de clase 1: predecir siempre "clase 0" ya daría ${(100 * (1 - data.nPos / data.train.n)).toFixed(0)}% de exactitud. Mira el <b>AP</b> (${fmt(m.ap, 3)}) y la curva Precisión–Recall en vez de la exactitud.`]);
    }
    if (state.noise > 0.05) {
      out.push(['Ruido de etiqueta', `Con un ${Math.round(state.noise * 100)}% de etiquetas cambiadas al azar, hay un <b>suelo</b> de error que ningún modelo puede bajar: si el logloss de entrenamiento cae muy por debajo del de validación, lo que está aprendiendo es ese ruido.`]);
    }
    $('#reading').innerHTML = out.map(([h, p]) => `<p><span class="r-h">${h}</span><br>${p}</p>`).join('');
  }

  function renderAll() {
    syncData(); syncParams(); renderTiles(); renderMap(); renderCurve(); renderImportance(); renderEval(); renderTree(); renderRead();
    writeHash();
  }

  /* =====================================================================
   *  Barrido
   * ===================================================================== */
  let sweepToken = 0, sweepState = null;
  const scheduleSweep = ML.debounce(() => runSweep(), 350);
  function trainOne(over) {
    const s = Object.assign({}, state, over);
    const t = data.train, v = data.val;
    const m = GBM.fit(t.X, t.y, t.n, 2, {
      nEstimators: Math.max(1, s.nEstimators), learningRate: s.lr, maxDepth: s.maxDepth,
      minChildWeight: s.minChildWeight, gamma: s.gamma, lambda: s.lambda, subsample: s.subsample, seed: s.seed,
    }, [{ X: v.X, y: v.y, n: v.n }]);
    const h = m.hist[m.hist.length - 1];
    const p = m.predictProba(v.X, v.n);
    const cv = M.curves(p, v.y);
    return { val: h.val0, train: h.train, gap: h.val0 - h.train, acc: h.val0Acc, auc: cv ? cv.auc : NaN, ap: cv ? cv.ap : NaN };
  }
  async function runSweep() {
    if (!data) return;
    const token = ++sweepToken, id = state.active, spec = SWEEPS[id];
    if (!spec) { $('#sweepChart').innerHTML = '<div class="empty">Este control no tiene barrido.</div>'; $('#sweepInsight').innerHTML = ''; return; }
    const results = [];
    sweepState = { id, spec, results, total: spec.values.length, running: true, current: state[id], token };
    drawSweep();
    for (const v of spec.values) {
      const metrics = { val: [], train: [], gap: [], acc: [], auc: [], ap: [] };
      for (let r = 0; r < spec.reps; r++) {
        const over = id === 'seed' ? { seed: v } : { [id]: v, seed: state.seed + r * 101 };
        const m = trainOne(over);
        Object.keys(metrics).forEach((k) => { if (Number.isFinite(m[k])) metrics[k].push(m[k]); });
      }
      if (token !== sweepToken) return;
      results.push({ x: v, metrics });
      drawSweep();
      await new Promise((r) => setTimeout(r, 0));
      if (token !== sweepToken) return;
    }
    sweepState.running = false;
    drawSweep();
  }
  function drawSweep() {
    const s = sweepState; if (!s) return;
    const met = state.sweepMetric;
    Ch.sweep($('#sweepChart'), { spec: s.spec, results: s.results, metric: met, metricLabel: SWEEP_METRICS[met], current: s.current, running: s.running, total: s.total });
    $('#sweepInsight').innerHTML = s.running ? '' : sweepInsight(met);
    const rows = s.results.map((r) => {
      const a = r.metrics[met]; if (!a.length) return '';
      const mu = a.reduce((x, v) => x + v, 0) / a.length, sd = Math.sqrt(a.reduce((x, v) => x + (v - mu) ** 2, 0) / a.length);
      return `<tr><td>${esc(String(s.spec.tickLabel ? s.spec.tickLabel(r.x) : r.x))}</td><td>${fmt(mu, 3)}</td><td>${fmt(sd, 3)}</td></tr>`;
    }).join('');
    $('#sweepTable').innerHTML = `<table><thead><tr><th>${esc(s.spec.code)}</th><th>media</th><th>desv.</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  function sweepInsight(met) {
    const s = sweepState; if (!s || s.results.length < 2) return '';
    const vals = s.results.map((r) => r.x);
    const st = s.results.map((r) => { const a = r.metrics[met]; const mu = a.reduce((x, v) => x + v, 0) / a.length; return { mu, sd: Math.sqrt(a.reduce((x, v) => x + (v - mu) ** 2, 0) / a.length) }; });
    const lower = !!LOWER_IS_BETTER[met];
    let bi = 0; st.forEach((x, i) => { if (lower ? x.mu < st[bi].mu : x.mu > st[bi].mu) bi = i; });
    const name = SWEEP_METRICS[met];
    const label = (v) => String(s.spec.tickLabel ? s.spec.tickLabel(v) : v);
    if (s.id === 'nEstimators' && met === 'val') {
      const rising = bi < st.length - 1 && st[st.length - 1].mu > st[bi].mu + 0.01;
      return rising
        ? `<b>El mínimo está en n_estimators ≈ ${label(vals[bi])}</b> (logloss ${fmt(st[bi].mu, 3)}) y a partir de ahí la validación <b>empeora</b>: ese es el punto que encontraría el <i>early stopping</i>. Más rondas aquí no es "más seguro": es sobreajuste.`
        : `En este rango la validación todavía baja hasta n_estimators = ${label(vals[vals.length - 1])}: aún no has llegado al punto de sobreajuste. Sube el ruido de etiqueta o la profundidad y repite el barrido para verlo aparecer.`;
    }
    if (s.id === 'lr') {
      return `<b>Mejor learning_rate ≈ ${label(vals[bi])}</b> (${name} ${fmt(st[bi].mu, 3)}) <i>con las ${state.nEstimators} rondas actuales</i>. Ese "mejor" se mueve si cambias n_estimators: con menos rondas conviene un η mayor, y con más rondas, uno menor. Por eso se ajustan juntos (capítulo 4).`;
    }
    if (s.id === 'maxDepth') {
      return `<b>Mejor max_depth = ${label(vals[bi])}</b> (${name} ${fmt(st[bi].mu, 3)}). ${bi < st.length - 1 ? 'Pasada esa profundidad el modelo empeora en validación: cada árbol memoriza más de lo que generaliza.' : 'En estos datos, más profundidad sigue ayudando: la frontera real es compleja.'}`;
    }
    if (s.id === 'gamma' || s.id === 'lambda' || s.id === 'minChildWeight') {
      return `<b>Mejor ${s.spec.code} = ${label(vals[bi])}</b> (${name} ${fmt(st[bi].mu, 3)}). Los tres frenos (gamma, lambda y min_child_weight) apuntan a lo mismo: menos varianza a cambio de algo de sesgo. Si los datos no tienen ruido, subirlos solo empeora.`;
    }
    if (s.id === 'seed') {
      const mus = st.map((x) => x.mu), mu = mus.reduce((a, b) => a + b, 0) / mus.length;
      return `Solo por cambiar la semilla, ${name} varía entre ${fmt(Math.min(...mus), 3)} y ${fmt(Math.max(...mus), 3)} (desv. ${fmt(Math.sqrt(mus.reduce((a, b) => a + (b - mu) ** 2, 0) / mus.length), 3)}). Diferencias menores que eso entre configuraciones <b>no son distinguibles</b>. ${state.subsample < 1 ? 'Con subsample < 1 la varianza es mayor.' : 'Con subsample = 1 la única fuente de azar son los datos.'}`;
    }
    return `Mejor valor del barrido: <b>${esc(label(vals[bi]))}</b> (${name} ${fmt(st[bi].mu, 3)}).`;
  }
  function renderSweepChrome() {
    const spec = SWEEPS[state.active];
    $('#sweepTitle').innerHTML = spec ? `Barrido de <code>${esc(spec.code)}</code>` : 'Barrido';
    $('#sweepHint').textContent = SWEEP_HINT[state.active] || '';
    $('#sweepMetricSeg').innerHTML = Object.keys(SWEEP_METRICS).map((m) => `<button type="button" data-m="${m}" aria-pressed="${m === state.sweepMetric}">${SWEEP_METRICS[m]}</button>`).join('');
    $$('#sweepMetricSeg button').forEach((b) => b.addEventListener('click', () => { state.sweepMetric = b.dataset.m; renderSweepChrome(); drawSweep(); }));
  }

  /* =====================================================================
   *  Escenarios
   * ===================================================================== */
  function buildLessons() {
    const host = $('#lessonChips');
    host.innerHTML = ML.GB_LESSONS.map((l) => `<button type="button" role="tab" data-id="${l.id}" aria-selected="false">${esc(l.title)}<small>${esc(l.tag)}</small></button>`).join('');
    $$('button', host).forEach((b) => b.addEventListener('click', () => applyLesson(b.dataset.id)));
    $('#lessonBody').innerHTML = '<p class="lesson-empty">Elige un escenario para cargar datos y parámetros y ver un concepto concreto.</p>';
  }
  function applyLesson(id) {
    const l = ML.GB_LESSONS.find((x) => x.id === id); if (!l) return;
    trainToken++; training = false; setTrainingUI(false);
    Object.assign(state, DEF, l.setup);
    state.active = l.active; state.open = l.active; state.lesson = id;
    $$('#lessonChips button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.id === id)));
    $('#lessonBody').innerHTML = `<div class="lb-title">${esc(l.title)}</div>
      <div><h4>Qué hacer</h4><ol>${l.steps.map((x) => `<li>${x}</li>`).join('')}</ol></div>
      <div><h4>Qué observar</h4><p>${l.look}</p></div>
      <div><button class="btn" type="button" id="lessonReset">↺ Reiniciar escenario</button></div>`;
    $('#lessonReset').addEventListener('click', () => applyLesson(id));
    buildData(); refit(); renderAll(); renderSweepChrome(); scheduleSweep();
    const t = document.getElementById(l.target) || $('#lab');
    if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  ML.applyLesson = applyLesson;

  /* =====================================================================
   *  URL y arranque
   * ===================================================================== */
  function writeHash() {
    const h = new URLSearchParams({
      ds: state.dataset, n: state.n, nz: Math.round(state.noise * 100), mi: state.minor,
      T: state.nEstimators, lr: state.lr, d: state.maxDepth, w: state.minChildWeight,
      g: state.gamma, la: state.lambda, sub: state.subsample, s: state.seed, p: state.active,
    });
    try { history.replaceState(null, '', '#' + h.toString()); } catch (e) { /* file:// */ }
  }
  function readHash() {
    if (!location.hash || location.hash.length < 4) return;
    const h = new URLSearchParams(location.hash.slice(1));
    if (!h.has('ds')) return;
    const num = (k, d) => (h.has(k) && Number.isFinite(+h.get(k)) ? +h.get(k) : d);
    if (D.SHAPES[h.get('ds')]) state.dataset = h.get('ds');
    state.n = [150, 300, 600].includes(num('n', 300)) ? num('n', 300) : 300;
    state.noise = clamp(num('nz', 0), 0, 30) / 100;
    state.minor = clamp(num('mi', 0.5), 0.02, 0.5);
    state.nEstimators = clamp(Math.round(num('T', 40)), 0, 300);
    state.lr = clamp(num('lr', 0.3), 0.01, 1);
    state.maxDepth = clamp(Math.round(num('d', 3)), 1, 8);
    state.minChildWeight = clamp(num('w', 1), 0.25, 40);
    state.gamma = clamp(num('g', 0), 0, 5);
    state.lambda = clamp(num('la', 1), 0, 20);
    state.subsample = clamp(num('sub', 1), 0.3, 1);
    state.seed = Math.max(0, Math.round(num('s', 1)));
    if (PMAP[h.get('p')]) { state.active = h.get('p'); state.open = state.active; }
  }

  function wireUI() {
    $('#togHeat').addEventListener('click', () => { state.showHeat = !state.showHeat; renderMap(); });
    $('#togPoints').addEventListener('click', () => { state.showPoints = !state.showPoints; renderMap(); });
    $('#treePrev').addEventListener('click', () => { state.treeIdx = Math.max(0, state.treeIdx - 1); renderTree(); });
    $('#treeNext').addEventListener('click', () => { state.treeIdx = Math.min(model.trees.length - 1, state.treeIdx + 1); renderTree(); });
    $$('#curveMetricSeg button').forEach((b) => b.addEventListener('click', () => { state.curveMetric = b.dataset.m; renderCurve(); }));
    $('#mapCanvas').addEventListener('click', (e) => {
      const r = e.target.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width, py = 1 - (e.clientY - r.top) / r.height;
      let best = -1, bd = Infinity;
      for (let i = 0; i < data.train.n; i++) {
        const d2 = (data.train.X[i * 2] - px) ** 2 + (data.train.X[i * 2 + 1] - py) ** 2;
        if (d2 < bd) { bd = d2; best = i; }
      }
      state.sel = bd < 0.002 ? best : -1;
      renderMap(); renderSelInfo();
    });
    window.addEventListener('themechange', () => { renderMap(); });
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => renderMap(), 150); });
  }
  function renderSelInfo() {
    const host = $('#selInfo'); if (!host) return;
    if (state.sel < 0) { host.innerHTML = '<span class="fine">Clic en un punto del mapa para ver su predicción.</span>'; return; }
    const i = state.sel, t = data.train;
    host.innerHTML = `<div class="kv"><span>Punto</span><b>#${i + 1}</b></div>
      <div class="kv"><span>Coordenadas</span><b class="mono">(${t.X[i * 2].toFixed(2)}, ${t.X[i * 2 + 1].toFixed(2)})</b></div>
      <div class="kv"><span>Etiqueta usada</span><b>clase ${t.y[i]}${t.yTrue[i] !== t.y[i] ? ' (¡ruido!)' : ''}</b></div>
      <div class="kv"><span>Probabilidad predicha</span><b>${fmt(trainP[i], 3)}</b></div>`;
  }

  function init() {
    ML.initTheme();
    readHash();
    buildData(); refit();
    buildParams(); buildDataControls(); buildLessons(); wireUI();
    renderAll(); renderSweepChrome(); renderSelInfo(); scheduleSweep();
    ML.gbLab = {
      state, get model() { return model; }, get data() { return data; }, get valP() { return valP; },
      metrics: metricsNow, bestRound, train, applyLesson, refit, renderAll,
    };
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
