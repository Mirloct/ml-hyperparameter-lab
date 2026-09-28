/* Capítulo 4: cuadrícula learning_rate × n_estimators.
 * Truco importante (el mismo que usa el early stopping): entrenar UNA vez con el n_estimators máximo
 * ya da el resultado de todas las rondas intermedias, porque los primeros T árboles no cambian.
 * Por eso la cuadrícula completa cuesta 5 entrenamientos, no 30. */
(function () {
  'use strict';
  const ML = window.MLLab;
  const GBM = ML.GBM, D = ML.gbData, M = ML.metrics;
  const { $, $$, clamp, fmt, esc } = ML;
  const H = ML.charts._h;

  const LRS = [0.02, 0.05, 0.1, 0.3, 0.9];
  const ROUNDS = [5, 10, 25, 50, 100, 200];
  const METRICS = {
    val: { label: 'Logloss de validación (menor mejor)', lower: true },
    auc: { label: 'ROC-AUC de validación (mayor mejor)', lower: false },
    gap: { label: 'Brecha train–val (menor mejor)', lower: true },
  };
  let cells = null, running = false;

  function run() {
    if (running) return;
    running = true;
    const btn = $('#hmRun'); btn.disabled = true;
    $('#hmState').textContent = 'entrenando 5 modelos (uno por learning_rate)…';
    setTimeout(() => {
      const shape = $('#hmDs').value, depth = +$('#hmDepth').value;
      const noise = +$('#hmNoise').value / 100;
      const d = D.generate(shape, 250, 12345, noise, 0.5);
      const t = d.train, v = d.val;
      cells = [];
      LRS.forEach((lr) => {
        const m = GBM.fit(t.X, t.y, t.n, 2, {
          nEstimators: ROUNDS[ROUNDS.length - 1], learningRate: lr, maxDepth: depth,
          minChildWeight: 1, gamma: 0, lambda: 1, subsample: 1, seed: 7,
        }, [{ X: v.X, y: v.y, n: v.n }]);
        const staged = m.stagedProba(v.X, v.n, ROUNDS);
        const row = ROUNDS.map((T, k) => {
          const h = m.hist[T - 1];
          const cv = M.curves(staged[k], v.y);
          return { lr, T, val: h.val0, train: h.train, gap: h.val0 - h.train, auc: cv ? cv.auc : NaN };
        });
        cells.push(row);
      });
      running = false; btn.disabled = false;
      $('#hmState').textContent = 'Listo. Cambia la métrica o la profundidad y vuelve a calcular.';
      draw();
    }, 30);
  }

  function draw() {
    const el = $('#hmChart');
    if (!cells) { el.innerHTML = '<div class="empty">Pulsa «Calcular cuadrícula» para entrenar los modelos.</div>'; $('#hmRead').innerHTML = ''; return; }
    const key = $('#hmMetric').value, spec = METRICS[key];
    const flat = cells.flat().map((c) => c[key]).filter(Number.isFinite);
    const lo = Math.min(...flat), hi = Math.max(...flat);
    const w = Math.max(280, el.clientWidth), cw = (w - 92) / ROUNDS.length, chh = 44;
    const hgt = chh * LRS.length + 64;
    const color = (v) => {
      const t = clamp((v - lo) / (hi - lo || 1), 0, 1);
      const good = spec.lower ? 1 - t : t;                       // 1 = mejor
      return `color-mix(in srgb, var(--s3) ${Math.round(12 + good * 72)}%, var(--surface))`;
    };
    let best = { v: spec.lower ? Infinity : -Infinity, lr: 0, T: 0 };
    cells.forEach((row) => row.forEach((c) => {
      const v = c[key];
      if (Number.isFinite(v) && (spec.lower ? v < best.v : v > best.v)) best = { v, lr: c.lr, T: c.T };
    }));
    let svg = `<svg class="ch" width="${w}" height="${hgt}" viewBox="0 0 ${w} ${hgt}" role="img" aria-label="Cuadrícula learning_rate por n_estimators">`;
    cells.forEach((row, r) => {
      svg += `<text x="84" y="${42 + r * chh + chh / 2 + 4}" text-anchor="end">${LRS[r]}</text>`;
      row.forEach((c, k) => {
        const x = 92 + k * cw, y = 42 + r * chh;
        const isBest = c.lr === best.lr && c.T === best.T;
        svg += `<rect x="${x + 1}" y="${y + 1}" width="${cw - 2}" height="${chh - 2}" rx="4" style="fill:${color(c[key])};stroke:${isBest ? 'var(--ink)' : 'var(--border)'};stroke-width:${isBest ? 2 : 1}"/>`;
        svg += `<text x="${x + cw / 2}" y="${y + chh / 2 + 4}" text-anchor="middle" style="fill:var(--ink)">${fmt(c[key], key === 'auc' ? 3 : 2)}</text>`;
      });
    });
    ROUNDS.forEach((T, k) => { svg += `<text x="${92 + k * cw + cw / 2}" y="34" text-anchor="middle">${T}</text>`; });
    svg += `<text class="title" x="${92 + (w - 92) / 2}" y="16" text-anchor="middle">n_estimators (nº de rondas) →</text>`;
    svg += `<text class="title" transform="translate(13 ${42 + (chh * LRS.length) / 2}) rotate(-90)" text-anchor="middle">learning_rate</text>`;
    svg += `<text class="title" x="${92 + (w - 92) / 2}" y="${hgt - 6}" text-anchor="middle">${esc(spec.label)}</text></svg>`;
    el.innerHTML = svg;

    // lectura: ¿dónde está el óptimo y qué forma tiene?
    const bestPerLr = cells.map((row) => {
      let b = row[0];
      row.forEach((c) => { if (spec.lower ? c[key] < b[key] : c[key] > b[key]) b = c; });
      return b;
    });
    const trend = bestPerLr.map((b) => b.T);
    const monotone = trend.every((v, i) => i === 0 || v <= trend[i - 1]);
    $('#hmRead').innerHTML = `<b>Mejor casilla:</b> learning_rate = ${best.lr} con n_estimators = ${best.T} (${spec.label.split(' (')[0]} ${fmt(best.v, 3)}). ` +
      `El óptimo de rondas por cada learning_rate es ${bestPerLr.map((b) => `η=${b.lr}→${b.T}`).join(', ')}. ` +
      (monotone
        ? 'Fíjate en la <b>diagonal</b>: cuanto menor es η, más rondas hacen falta para llegar al mismo punto. Ese es el sentido de "se ajustan juntos": preguntar "¿cuántas rondas?" sin decir con qué η no tiene respuesta.'
        : 'El óptimo de rondas no cae en una diagonal perfecta porque el ruido entre configuraciones es del mismo orden que las diferencias: con datos ruidosos, repite con varias semillas antes de declarar un ganador.') +
      ' <span class="fine">Esta cuadrícula se calculó con 5 entrenamientos, no 30: entrenar una vez con 200 rondas ya contiene los resultados de 5, 10, 25… rondas. Es exactamente lo que aprovecha el <i>early stopping</i>.</span>';
  }

  function init() {
    const sel = $('#hmDs');
    if (!sel) return;
    sel.innerHTML = Object.entries(D.SHAPES).sort((a, b) => a[1].order - b[1].order)
      .map(([id, d]) => `<option value="${id}"${id === 'moons' ? ' selected' : ''}>${esc(d.name)}</option>`).join('');
    $('#hmRun').addEventListener('click', run);
    $('#hmMetric').addEventListener('change', draw);
    draw();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
