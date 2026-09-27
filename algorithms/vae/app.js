/* Controlador del laboratorio de VAE: estado, entrenamiento por tramos y renderizado. */
(function () {
  'use strict';
  const ML = window.MLLab;
  const VAE = ML.VAE, SH = ML.shapes, M = ML.metrics, U = ML.unsup, Ch = ML.charts;
  const { $, $$, clamp, fmt, esc } = ML;
  const PARAMS = ML.VAE_PARAMS, PMAP = Object.fromEntries(PARAMS.map((p) => [p.id, p]));

  const DEF = { dataset: 'dot', n: 400, anomalies: 0, noise: 0.02, latent: 2, beta: 1, hidden: 24, lr: 0.01, warmup: 0, batch: 16, seed: 1, epochs: 0 };
  const STOPS = {
    latent: [1, 2, 3, 4, 6, 8, 12, 16],
    beta: [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16],
    hidden: [0, 4, 8, 12, 16, 24, 32, 48, 64],
    lr: [0.0003, 0.001, 0.003, 0.01, 0.02, 0.03, 0.1, 0.3],
    warmup: [0, 5, 10, 15, 25, 40],
    batch: [4, 8, 16, 32, 64, 128],
  };
  const SWEEPS = {
    latent: { code: 'latent_dim', kind: 'log', values: [1, 2, 3, 4, 6, 8, 12, 16], axisLabel: 'latent_dim (escala log)', reps: 2 },
    beta: { code: 'beta', kind: 'lin', values: [0, 0.5, 1, 2, 3, 4, 6, 8], axisLabel: 'β (peso del KL)', reps: 2 },
    hidden: { code: 'hidden_units', kind: 'log', values: [4, 8, 12, 16, 24, 32, 48, 64], axisLabel: 'unidades ocultas (escala log)', reps: 2 },
    lr: { code: 'learning_rate', kind: 'log', values: [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3], tickLabel: (v) => String(v), axisLabel: 'learning_rate (escala log)', reps: 2 },
    warmup: { code: 'kl_warmup', kind: 'lin', values: [0, 5, 10, 15, 25, 40], axisLabel: 'épocas de recocido del KL', reps: 2 },
    batch: { code: 'batch_size', kind: 'log', values: [4, 8, 16, 32, 64, 128], axisLabel: 'batch_size (escala log)', reps: 2 },
    epochs: { code: 'epochs', kind: 'log', values: [5, 10, 20, 40, 70, 110, 160], axisLabel: 'épocas (escala log)', reps: 2 },
    seed: { code: 'random_state', kind: 'cat', values: [1, 2, 3, 4, 5, 6, 7, 8], band: true, axisLabel: 'random_state (cada punto = una semilla)', reps: 1 },
  };
  const SWEEP_HINT = {
    latent: 'Cada valor se entrena desde cero (2 semillas). Fíjate en dónde deja de mejorar la reconstrucción.',
    beta: 'Observa la caída: pasado cierto β el KL se va a cero y la reconstrucción se estanca en su peor valor.',
    hidden: 'Más capacidad mejora hasta aplanarse; muy poca provoca colapso por falta de capacidad.',
    lr: 'La curva en U típica: demasiado baja no llega, demasiado alta diverge.',
    warmup: 'Solo se nota si β es alto: ahí decide entre colapsar o no.',
    batch: 'Interactúa con la tasa de aprendizaje; el efecto suele ser menor que el de β o η.',
    epochs: 'Entrenar más baja la pérdida hasta el plateau. No es un hiperparámetro a optimizar.',
    seed: 'Cada punto es el mismo modelo con otra semilla: su dispersión es la varianza atribuible al azar.',
  };
  const SWEEP_METRICS = { rec: 'Reconstrucción', kl: 'KL', elbo: 'Pérdida (−ELBO)', active: 'Dimensiones activas', dis: 'Alineación con factores', ap: 'AP (anomalías)' };

  const state = { ...DEF, active: 'beta', open: 'beta', sweepMetric: 'rec', sel: 0, genDimX: 0, genDimY: 1, colorBy: 0, lesson: null, showRecon: true };
  let data = null, model = null, ev = null, training = false, trainToken = 0;

  /* =====================================================================
   *  Datos y modelo
   * ===================================================================== */
  function buildData() {
    data = SH.generate(state.dataset, state.n, 7, state.anomalies, state.noise);
    state.nF = data.nF;
    state.sel = Math.min(state.sel, data.n - 1);
  }
  function newModel() {
    model = new VAE({ D: data.D, latent: state.latent, hidden: state.hidden, seed: state.seed, lr: state.lr, beta: state.beta });
    ev = model.evaluate(data.X, data.n);
    state.genDimX = 0; state.genDimY = Math.min(1, state.latent - 1);
  }
  /** Entrena `k` épocas sin bloquear la interfaz (troceado con requestAnimationFrame). */
  async function train(k) {
    if (training) return;
    training = true; const token = ++trainToken;
    setTrainingUI(true);
    const chunk = 5;
    for (let done = 0; done < k; done += chunk) {
      const m = Math.min(chunk, k - done);
      for (let i = 0; i < m; i++) model.trainEpoch(data.X, data.n, state.batch, VAE.annealed(state.beta, model.epoch + 1, state.warmup));
      state.epochs = model.epoch;
      ev = model.evaluate(data.X, data.n);
      renderAll();
      await new Promise((r) => requestAnimationFrame(r));
      if (token !== trainToken) { training = false; return; }
    }
    training = false; setTrainingUI(false);
    renderAll(); scheduleSweep();
  }
  function setTrainingUI(on) {
    $$('.train-btn').forEach((b) => { b.disabled = on; });
    $('#trainState').textContent = on ? 'entrenando…' : '';
  }
  function reset() { trainToken++; training = false; setTrainingUI(false); newModel(); state.epochs = 0; renderAll(); scheduleSweep(); }

  /* =====================================================================
   *  Métricas
   * ===================================================================== */
  /** Alineación latente ↔ factores reales: por cada factor, la mayor |correlación de rangos|. */
  function alignment() {
    if (!data.nF) return { mean: NaN, per: [] };
    const per = [];
    for (let f = 0; f < data.nF; f++) {
      const fv = Float64Array.from({ length: data.n }, (_, i) => data.F[i * data.nF + f]);
      let best = 0, bj = -1;
      for (let j = 0; j < state.latent; j++) {
        const z = Float64Array.from({ length: data.n }, (_, i) => ev.mu[i * state.latent + j]);
        const c = Math.abs(U.spearman(fv, z));
        if (c > best) { best = c; bj = j; }
      }
      per.push({ name: data.factors[f], c: best, dim: bj });
    }
    return { mean: per.reduce((s, p) => s + p.c, 0) / per.length, per };
  }
  function anomalyMetrics() {
    if (!state.anomalies) return null;
    const cv = M.curves(ev.recErr, data.y);
    return cv ? { auc: cv.auc, ap: cv.ap, cv } : null;
  }
  function metricsNow() {
    const a = alignment(), an = anomalyMetrics();
    return { rec: ev.rec, kl: ev.kl, elbo: ev.elbo, active: ev.active, dis: a.mean, auc: an ? an.auc : NaN, ap: an ? an.ap : NaN };
  }

  /* =====================================================================
   *  Controles
   * ===================================================================== */
  function fmtParam(id, v = state[id]) {
    if (id === 'lr') return String(v);
    if (id === 'beta') return String(v);
    if (id === 'hidden') return v === 0 ? 'lineal' : String(v);
    if (id === 'warmup') return v === 0 ? 'sin recocido' : v + ' ép.';
    if (id === 'epochs') return String(model ? model.epoch : 0);
    return String(v);
  }
  /** Cambiar un hiperparámetro del modelo obliga a reentrenar desde cero. */
  function setParam(id, v) {
    if (state[id] === v) return;
    state[id] = v;
    openParam(id, false);
    if (id === 'batch' || id === 'warmup') { syncParams(); scheduleSweep(); return; }  // no requieren reconstruir
    reset();
  }
  function openParam(id, toggle) {
    const changed = state.active !== id;
    state.active = id;
    state.open = toggle && state.open === id ? null : id;
    $$('.pcard').forEach((el) => el.classList.toggle('active', el.dataset.id === state.open));
    if (changed) { state.sweepMetric = id === 'beta' ? 'kl' : 'rec'; renderSweepChrome(); scheduleSweep(); }
  }
  function controlHTML(p) {
    if (p.id === 'epochs') {
      return `<div class="ep-row">
        <button class="btn sm primary train-btn" type="button" data-ep="20">+20 épocas</button>
        <button class="btn sm train-btn" type="button" data-ep="50">+50</button>
        <button class="btn sm train-btn" type="button" data-ep="100">+100</button>
        <button class="btn sm" type="button" data-reset>↺ Reiniciar</button>
        <span class="train-state" id="trainState"></span></div>`;
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
      if (p.id === 'epochs') {
        $$('[data-ep]', el).forEach((b) => b.addEventListener('click', () => train(+b.dataset.ep)));
        el.querySelector('[data-reset]').addEventListener('click', reset);
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
      if (STOPS[p.id]) { const i = STOPS[p.id].indexOf(state[p.id]); el.querySelector('input[type=range]').value = i >= 0 ? i : 0; }
      if (p.id === 'seed') { const inp = el.querySelector('input'); if (document.activeElement !== inp) inp.value = state.seed; }
      el.querySelector('[data-live]').innerHTML = p.live(state, model, ev);
    });
  }

  function buildDataControls() {
    const list = $('#dsList');
    list.innerHTML = Object.entries(SH.DEFS).map(([id, d]) => `<button type="button" role="radio" data-id="${id}">${esc(d.name)}</button>`).join('');
    $$('button', list).forEach((b) => b.addEventListener('click', () => { state.dataset = b.dataset.id; buildData(); reset(); }));
    $('#nSel').addEventListener('change', (e) => { state.n = +e.target.value; buildData(); reset(); });
    $('#anomSel').addEventListener('change', (e) => { state.anomalies = +e.target.value; buildData(); reset(); });
    $('#noiseRange').addEventListener('input', (e) => { state.noise = +e.target.value / 100; $('#noiseVal').textContent = (+e.target.value) + '%'; buildData(); reset(); });
  }
  function syncData() {
    $$('#dsList button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.id === state.dataset)));
    $('#dsBlurb').innerHTML = SH.DEFS[state.dataset].blurb + ` <b>Factores reales: ${data.factors.map(esc).join(', ')}</b> (${data.nF}).`;
    $('#nSel').value = String(state.n); $('#anomSel').value = String(state.anomalies);
    $('#noiseRange').value = state.noise * 100; $('#noiseVal').textContent = Math.round(state.noise * 100) + '%';
  }

  /* =====================================================================
   *  Render
   * ===================================================================== */
  function renderTiles() {
    const m = metricsNow();
    const groups = { fit: ['rec', 'kl', 'elbo', 'active'], fact: ['dis'], anom: ['auc', 'ap'] };
    Object.entries(groups).forEach(([g, ids]) => {
      const host = $('#tiles-' + g); if (!host) return;
      host.innerHTML = ids.map((id) => {
        const d = ML.VAE_METRICS[id], v = m[id];
        const txt = id === 'active' ? `${v}<small> / ${state.latent}</small>` : (Number.isFinite(v) ? fmt(v, id === 'rec' || id === 'elbo' ? 2 : 3) : '—');
        return `<div class="tile" tabindex="0" data-tip="${esc(d.tip)}"><div class="t-label">${d.label}<i>?</i></div><div class="t-val">${txt}</div></div>`;
      }).join('');
    });
    $('#anomGroup').hidden = !state.anomalies;
    $('#miniMetrics').innerHTML = [['Reconstrucción', fmt(m.rec, 1)], ['KL', fmt(m.kl, 2)], ['Activas', `${m.active}/${state.latent}`], ['Épocas', model.epoch]]
      .map(([k, v]) => `<div class="mm"><span>${k}</span><b>${v}</b></div>`).join('');
  }

  function renderImages() {
    const cols = 8, k = cols * 2;
    const idx = [];
    for (let i = 0; i < k && i < data.n; i++) idx.push(Math.floor((i * data.n) / k));
    if (state.anomalies) { for (let i = 0; i < data.n; i++) if (data.y[i] && idx.length < k + 4) idx.push(i); }
    const imgs = [], labels = [], marks = [];
    const c = ML.colors();
    idx.forEach((i) => { imgs.push(data.X.subarray(i * data.D, (i + 1) * data.D)); labels.push(data.y[i] ? 'anómalo' : ''); marks.push(data.y[i] ? c.s2 : null); });
    Ch.imageGrid($('#origGrid'), { imgs, S: data.S, cols: Math.min(cols, idx.length), labels, marks });
    const rec = idx.map((i) => ev.recon.subarray(i * data.D, (i + 1) * data.D));
    Ch.imageGrid($('#reconGrid'), { imgs: rec, S: data.S, cols: Math.min(cols, idx.length), marks });
  }

  /** Mapa generativo: decodifica una rejilla del espacio latente (los demás ejes en 0). */
  function renderGen() {
    const G = state.latent === 1 ? 1 : 9, span = 2.2;
    const imgs = [];
    const z = new Float64Array(state.latent);
    if (state.latent === 1) {
      for (let i = 0; i < 9; i++) { z[0] = -span + (2 * span * i) / 8; imgs.push(model.generate(z)); }
      Ch.imageGrid($('#genGrid'), { imgs, S: data.S, cols: 9 });
    } else {
      for (let r = 0; r < G; r++) for (let cI = 0; cI < G; cI++) {
        z.fill(0);
        z[state.genDimX] = -span + (2 * span * cI) / (G - 1);
        z[state.genDimY] = span - (2 * span * r) / (G - 1);
        imgs.push(model.generate(z));
      }
      Ch.imageGrid($('#genGrid'), { imgs, S: data.S, cols: G, gap: 2 });
    }
    $('#genAxes').innerHTML = state.latent === 1 ? 'Recorriendo z1 de −2.2 a +2.2.' :
      `Horizontal: <b>z${state.genDimX + 1}</b> · Vertical: <b>z${state.genDimY + 1}</b>, de −2.2 a +2.2. Las demás dimensiones en 0.`;
  }

  function renderLatent() {
    if (state.latent < 2) { $('#latentChart').innerHTML = '<div class="empty">Con latent_dim = 1 no hay plano que dibujar: sube la dimensión latente a 2.</div>'; return; }
    const color = data.nF ? Float64Array.from({ length: data.n }, (_, i) => data.F[i * data.nF + state.colorBy]) : null;
    Ch.latentScatter($('#latentChart'), {
      mu: ev.mu, L: state.latent, dimX: state.genDimX, dimY: state.genDimY,
      color, colorLabel: data.factors[state.colorBy], y: state.anomalies ? data.y : null, sel: state.sel,
      onPick: (i) => { state.sel = i; renderSel(); },
    });
  }

  function renderSel() {
    const i = state.sel, off = i * data.D;
    Ch.imageGrid($('#selGrid'), { imgs: [data.X.subarray(off, off + data.D), ev.recon.subarray(off, off + data.D)], S: data.S, cols: 2, labels: ['original', 'reconstruida'] });
    const z = Array.from({ length: state.latent }, (_, j) => ev.mu[i * state.latent + j]);
    $('#selInfo').innerHTML = `<div class="kv"><span>Observación</span><b>#${i + 1}${data.y[i] ? ' · anómala' : ''}</b></div>
      <div class="kv"><span>Error de reconstrucción</span><b>${fmt(ev.recErr[i], 1)} nats</b></div>
      <div class="kv"><span>KL de esta observación</span><b>${fmt(ev.klPer[i], 2)} nats</b></div>
      <div class="kv"><span>Código z</span><b class="mono">[${z.map((v) => v.toFixed(2)).join(', ')}]</b></div>`;
  }

  function renderFactors() {
    const a = alignment();
    if (!a.per.length) { $('#factTable').innerHTML = ''; return; }
    $('#factTable').innerHTML = `<table class="fact-table"><thead><tr><th>Factor real</th><th>Mejor dimensión</th><th>|correlación|</th><th></th></tr></thead><tbody>` +
      a.per.map((p) => `<tr><td>${esc(p.name)}</td><td>z${p.dim + 1}</td><td>${fmt(p.c, 2)}</td>
        <td><div class="cbar"><i style="width:${Math.round(p.c * 100)}%;background:${p.c > 0.8 ? 'var(--s3)' : p.c > 0.5 ? 'var(--s4)' : 'var(--s2)'}"></i></div></td></tr>`).join('') +
      `</tbody></table><p class="fine">Cerca de 1 = ese factor vive en una sola dimensión latente (desenredado). Dos factores en la <b>misma</b> dimensión significa que están enredados. Esto solo se puede medir porque los datos son sintéticos.</p>`;
  }

  function renderAnom() {
    const wrap = $('#card-anom');
    wrap.hidden = !state.anomalies;
    if (!state.anomalies) return;
    const an = anomalyMetrics();
    const k = Math.max(1, Math.round(data.n * state.anomalies));
    const sorted = Array.from(ev.recErr).sort((a, b) => b - a);
    const thr = sorted[k - 1];
    Ch.errHist($('#errChart'), { err: ev.recErr, y: data.y, thr, hasBoth: true });
    const conf = M.confusion(ev.recErr, data.y, thr);
    $('#anomRead').innerHTML = an ? `Usando el error de reconstrucción como score: <b>ROC-AUC ${fmt(an.auc, 3)}</b> · <b>AP ${fmt(an.ap, 3)}</b>. Marcando las ${k} peores reconstrucciones, la precisión es ${fmt(conf.precision, 2)} y el recall ${fmt(conf.recall, 2)}. ${ev.kl < 0.05 ? '<b>⚠ El modelo está colapsado</b> (KL ≈ 0): sigue dando scores, pero ya no representa nada.' : ''}` : '';
  }

  function renderRead() {
    const m = metricsNow(), out = [];
    if (model.epoch === 0) { $('#reading').innerHTML = '<p><span class="r-h">Sin entrenar</span><br>El modelo está recién inicializado: sus reconstrucciones son una mancha gris. Pulsa <b>+20 épocas</b> para empezar.</p>'; return; }
    out.push(['Ajuste', `Tras <b>${model.epoch}</b> épocas: reconstrucción <b>${fmt(m.rec, 2)}</b> nats y KL <b>${fmt(m.kl, 2)}</b> nats. ${m.active} de ${state.latent} dimensiones latentes están activas.`]);
    if (m.kl < 0.05) out.push(['⚠ Colapso posterior', `El KL es prácticamente cero: el modelo <b>ignora el código latente</b> y reconstruye siempre lo mismo. Causas habituales: β demasiado alto (ahora ${state.beta}), red demasiado pequeña (ahora ${state.hidden || 'lineal'}) o falta de recocido. Prueba a activar el <b>recocido del KL</b> o bajar β.`]);
    else if (m.kl > 20) out.push(['KL muy alto', `El latente transporta mucha información pero se aleja del prior: el espacio queda <b>disperso y con huecos</b>, y muestrear del prior producirá datos poco realistas. Es lo que ocurre con β cercano a 0.`]);
    if (data.nF) {
      const a = alignment(), weak = a.per.filter((p) => p.c < 0.5);
      out.push(['Factores', weak.length
        ? `El modelo <b>no está codificando</b> ${weak.map((p) => `«${esc(p.name)}»`).join(' ni ')} (correlación ${weak.map((p) => fmt(p.c, 2)).join(', ')}). ${state.latent < data.nF ? `Tienes ${state.latent} dimensiones para ${data.nF} factores: no le caben.` : 'Hay dimensiones suficientes, así que revisa β y las épocas.'}`
        : `Los ${data.nF} factores reales están representados (alineación media ${fmt(a.mean, 2)}).`]);
    }
    const last = model.hist.slice(-5);
    if (last.length === 5) {
      const d = last[0].rec - last[4].rec;
      out.push(['¿Sigue aprendiendo?', Math.abs(d) < 0.05
        ? 'La reconstrucción cambió menos de 0.05 nats en las últimas 5 épocas: estás en el <b>plateau</b>, más épocas no ayudarán.'
        : `La reconstrucción bajó ${fmt(d, 2)} nats en las últimas 5 épocas: <b>todavía está aprendiendo</b>.`]);
    }
    $('#reading').innerHTML = out.map(([h, p]) => `<p><span class="r-h">${h}</span><br>${p}</p>`).join('');
  }

  function renderAll() {
    syncData(); syncParams(); renderTiles(); renderImages(); renderGen(); renderLatent(); renderSel(); renderFactors(); renderAnom(); renderRead();
    Ch.trainCurve($('#curveChart'), { hist: model.hist, beta: state.beta, running: training, warmup: state.warmup });
    Ch.klBars($('#klChart'), { klDim: ev.klDim, muVar: ev.muVar });
    renderDimPickers();
    writeHash();
  }
  function renderDimPickers() {
    const opts = Array.from({ length: state.latent }, (_, j) => `<option value="${j}">z${j + 1}</option>`).join('');
    ['#dimX', '#dimY'].forEach((s, k) => { const el = $(s); el.innerHTML = opts; el.value = String(k ? state.genDimY : state.genDimX); el.disabled = state.latent < 2; });
    const cb = $('#colorBy');
    cb.innerHTML = data.factors.map((f, j) => `<option value="${j}">${esc(f)}</option>`).join('');
    cb.value = String(state.colorBy);
  }

  /* =====================================================================
   *  Barrido (entrena varios modelos desde cero)
   * ===================================================================== */
  let sweepToken = 0, sweepState = null;
  const scheduleSweep = ML.debounce(() => runSweep(), 400);
  function trainOne(over) {
    const s = { ...state, ...over };
    const m = new VAE({ D: data.D, latent: s.latent, hidden: s.hidden, seed: s.seed, lr: s.lr, beta: s.beta });
    const E = Math.max(5, s.epochs || 40);
    for (let e = 0; e < E; e++) m.trainEpoch(data.X, data.n, s.batch, VAE.annealed(s.beta, e + 1, s.warmup));
    const v = m.evaluate(data.X, data.n);
    let dis = NaN;
    if (data.nF) {
      let sum = 0;
      for (let f = 0; f < data.nF; f++) {
        const fv = Float64Array.from({ length: data.n }, (_, i) => data.F[i * data.nF + f]);
        let best = 0;
        for (let j = 0; j < s.latent; j++) best = Math.max(best, Math.abs(U.spearman(fv, Float64Array.from({ length: data.n }, (_, i) => v.mu[i * s.latent + j]))));
        sum += best;
      }
      dis = sum / data.nF;
    }
    const ap = state.anomalies ? (M.curves(v.recErr, data.y) || {}).ap : NaN;
    return { rec: v.rec, kl: v.kl, elbo: v.elbo, active: v.active, dis, ap };
  }
  async function runSweep() {
    if (!data) return;
    const token = ++sweepToken, id = state.active, spec = SWEEPS[id];
    if (!spec) { $('#sweepChart').innerHTML = '<div class="empty">Este control no tiene barrido.</div>'; $('#sweepInsight').innerHTML = ''; return; }
    const results = [];
    sweepState = { id, spec, results, total: spec.values.length, running: true, current: state[id], token };
    drawSweep();
    for (const v of spec.values) {
      const metrics = { rec: [], kl: [], elbo: [], active: [], dis: [], ap: [] };
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
    const met = s.results.length && !s.results[0].metrics[state.sweepMetric].length ? 'rec' : state.sweepMetric;
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
    const name = SWEEP_METRICS[met];
    if (s.id === 'beta') {
      const kls = s.results.map((r) => r.metrics.kl.reduce((x, v) => x + v, 0) / r.metrics.kl.length);
      const ci = kls.findIndex((k) => k < 0.05);
      return ci > 0 ? `<b>Colapso a partir de β ≈ ${vals[ci]}:</b> el KL cae por debajo de 0.05 nats y el modelo deja de usar el latente. Por debajo de ese valor, subir β ordena el espacio; por encima, lo destruye. Prueba a activar el <b>recocido</b> y repite el barrido.`
        : `En este rango el KL nunca llega a cero: el modelo aguanta β hasta ${vals[vals.length - 1]} sin colapsar. Con redes más potentes el colapso aparece antes.`;
    }
    if (s.id === 'lr') {
      let bi = 0; st.forEach((x, i) => { if (x.mu < st[bi].mu) bi = i; });
      return `<b>Mejor η ≈ ${vals[bi]}</b> (${name} ${fmt(st[bi].mu, 2)}). Con η = ${vals[0]} el modelo apenas aprende y con η = ${vals[vals.length - 1]} ${st[st.length - 1].mu > st[bi].mu * 2 ? 'se desestabiliza' : 'empeora'}: la curva en U típica. Por eso se barre en escala logarítmica.`;
    }
    if (s.id === 'latent') {
      const act = s.results.map((r) => r.metrics.active.reduce((x, v) => x + v, 0) / r.metrics.active.length);
      const mx = Math.max(...act), sat = vals[act.findIndex((a) => a >= mx - 0.25)];
      return `Las dimensiones <b>activas</b> se estabilizan en ≈ <b>${fmt(mx, 1)}</b> a partir de latent_dim = ${sat}, aunque le des más: el VAE apaga lo que no necesita. Los datos tienen ${data.nF} factores reales.`;
    }
    if (s.id === 'seed') {
      const mus = st.map((x) => x.mu), mu = mus.reduce((a, b) => a + b, 0) / mus.length;
      return `Solo por cambiar la semilla, ${name} varía entre ${fmt(Math.min(...mus), 2)} y ${fmt(Math.max(...mus), 2)} (desv. ${fmt(Math.sqrt(mus.reduce((a, b) => a + (b - mu) ** 2, 0) / mus.length), 3)}). Diferencias menores que eso entre configuraciones <b>no son distinguibles</b>.`;
    }
    let bi = 0; st.forEach((x, i) => { if (x.mu < st[bi].mu) bi = i; });
    const last = st.length - 1;
    return `Mejor valor del barrido: <b>${esc(String(s.spec.tickLabel ? s.spec.tickLabel(vals[bi]) : vals[bi]))}</b> (${name} ${fmt(st[bi].mu, 2)}). ${Math.abs(st[last].mu - st[bi].mu) < 0.1 ? 'La mejora se aplana al final del rango: ahí está el <b>plateau</b>.' : ''}`;
  }
  function renderSweepChrome() {
    const spec = SWEEPS[state.active];
    $('#sweepTitle').innerHTML = spec ? `Barrido de <code>${esc(spec.code)}</code>` : 'Barrido';
    $('#sweepHint').textContent = SWEEP_HINT[state.active] || '';
    const avail = ['rec', 'kl', 'elbo', 'active'].concat(data && data.nF ? ['dis'] : []).concat(state.anomalies ? ['ap'] : []);
    $('#sweepMetricSeg').innerHTML = avail.map((m) => `<button type="button" data-m="${m}" aria-pressed="${m === state.sweepMetric}">${SWEEP_METRICS[m]}</button>`).join('');
    $$('#sweepMetricSeg button').forEach((b) => b.addEventListener('click', () => { state.sweepMetric = b.dataset.m; renderSweepChrome(); drawSweep(); }));
  }

  /* =====================================================================
   *  Escenarios
   * ===================================================================== */
  function buildLessons() {
    const host = $('#lessonChips');
    host.innerHTML = ML.VAE_LESSONS.map((l) => `<button type="button" role="tab" data-id="${l.id}" aria-selected="false">${esc(l.title)}<small>${esc(l.tag)}</small></button>`).join('');
    $$('button', host).forEach((b) => b.addEventListener('click', () => applyLesson(b.dataset.id)));
    $('#lessonBody').innerHTML = '<p class="lesson-empty">Elige un escenario para cargar datos y parámetros y ver un concepto concreto.</p>';
  }
  async function applyLesson(id) {
    const l = ML.VAE_LESSONS.find((x) => x.id === id); if (!l) return;
    trainToken++; training = false; setTrainingUI(false);
    Object.assign(state, DEF, l.setup);
    state.active = l.active; state.open = l.active; state.lesson = id;
    state.sweepMetric = l.active === 'beta' ? 'kl' : 'rec';
    $$('#lessonChips button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.id === id)));
    $('#lessonBody').innerHTML = `<div class="lb-title">${esc(l.title)}</div>
      <div><h4>Qué hacer</h4><ol>${l.steps.map((x) => `<li>${x}</li>`).join('')}</ol></div>
      <div><h4>Qué observar</h4><p>${l.look}</p></div>
      <div><button class="btn" type="button" id="lessonReset">↺ Reiniciar escenario</button></div>`;
    $('#lessonReset').addEventListener('click', () => applyLesson(id));
    buildData(); newModel(); renderAll(); renderSweepChrome();
    $('#lab').scrollIntoView({ behavior: 'smooth', block: 'start' });
    const ep = l.setup.epochs || 0;
    if (ep > 0) await train(ep); else scheduleSweep();
  }
  ML.applyLesson = applyLesson;

  /* =====================================================================
   *  URL y arranque
   * ===================================================================== */
  function writeHash() {
    const h = new URLSearchParams({ ds: state.dataset, n: state.n, an: state.anomalies, nz: Math.round(state.noise * 100), L: state.latent, b: state.beta, H: state.hidden, lr: state.lr, w: state.warmup, bs: state.batch, s: state.seed, e: model.epoch, p: state.active });
    try { history.replaceState(null, '', '#' + h.toString()); } catch (e) { /* file:// */ }
  }
  function readHash() {
    if (!location.hash || location.hash.length < 4) return 0;
    const h = new URLSearchParams(location.hash.slice(1));
    if (!h.has('ds')) return 0;
    const num = (k, d) => (h.has(k) && Number.isFinite(+h.get(k)) ? +h.get(k) : d);
    if (SH.DEFS[h.get('ds')]) state.dataset = h.get('ds');
    state.n = [200, 400, 800].includes(num('n', 400)) ? num('n', 400) : 400;
    state.anomalies = clamp(num('an', 0), 0, 0.2); state.noise = clamp(num('nz', 2), 0, 20) / 100;
    state.latent = clamp(Math.round(num('L', 2)), 1, 16); state.beta = clamp(num('b', 1), 0, 16);
    state.hidden = clamp(Math.round(num('H', 24)), 0, 64); state.lr = clamp(num('lr', 0.01), 0.0001, 0.5);
    state.warmup = clamp(Math.round(num('w', 0)), 0, 40); state.batch = clamp(Math.round(num('bs', 16)), 4, 128);
    state.seed = Math.max(0, Math.round(num('s', 1)));
    if (PMAP[h.get('p')]) { state.active = h.get('p'); state.open = state.active; }
    return clamp(Math.round(num('e', 0)), 0, 400);
  }

  function wireUI() {
    $('#dimX').addEventListener('change', (e) => { state.genDimX = +e.target.value; renderGen(); renderLatent(); });
    $('#dimY').addEventListener('change', (e) => { state.genDimY = +e.target.value; renderGen(); renderLatent(); });
    $('#colorBy').addEventListener('change', (e) => { state.colorBy = +e.target.value; renderLatent(); });
    $$('[data-pick]').forEach((b) => b.addEventListener('click', () => {
      const o = Array.from(ev.recErr.keys()).sort((a, c) => ev.recErr[c] - ev.recErr[a]);
      state.sel = b.dataset.pick === 'worst' ? o[0] : b.dataset.pick === 'best' ? o[o.length - 1] : o[Math.floor(o.length / 2)];
      renderSel(); renderLatent();
    }));
    window.addEventListener('themechange', () => { renderImages(); renderGen(); renderSel(); });
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { renderImages(); renderGen(); renderSel(); }, 150); });
  }

  async function init() {
    ML.initTheme();
    const ep = readHash();
    buildData(); newModel();
    buildParams(); buildDataControls(); buildLessons(); wireUI();
    renderAll(); renderSweepChrome();
    ML.vaeLab = { state, get model() { return model; }, get data() { return data; }, get ev() { return ev; }, train, applyLesson };
    await train(ep > 0 ? ep : 40);   // un modelo ya entrenado se entiende mucho mejor que uno vacío
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
