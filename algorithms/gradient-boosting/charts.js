/* Gráficas propias del laboratorio de Gradient Boosting (SVG y canvas a mano, sin librerías).
 * Reutiliza los primitivos y las gráficas genéricas de ../isolation-forest/charts.js
 * (C.sweep, C.curve para ROC/PR y C.scoreHist para el histograma de probabilidades). */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});
  const { clamp, fmt, esc } = ML;
  const C = ML.charts, H = C._h;
  const { lin, niceTicks, box, svgOpen, gridY, axisX, xTitle, yTitle, hover, dot, row } = H;

  /* ---------- 1. Mapa de decisión (canvas): probabilidad en todo el plano + puntos ---------- */
  /** grid: Float64Array(G*G) con la probabilidad predicha, recorriendo x1 de 0→1 y x2 de 1→0. */
  C.decisionMap = function (cv, { grid, G, X, y, yTrue, n, pred, showPoints = true, showHeat = true, sel = -1 }) {
    const c = ML.colors();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const W = Math.max(200, cv.clientWidth), Hh = W;
    cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr);
    cv.style.height = Hh + 'px';
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, Hh);

    // Fondo: mapa de calor divergente (clase 0 → azul s1, clase 1 → naranja s2, frontera ≈ blanco/gris)
    if (showHeat) {
      const cell = W / G;
      for (let r = 0; r < G; r++) {
        for (let k = 0; k < G; k++) {
          const p = grid[r * G + k];
          const t = clamp(Math.abs(p - 0.5) * 2, 0, 1);            // 0 = frontera, 1 = certeza
          const base = p >= 0.5 ? [235, 104, 52] : [42, 120, 214]; // s2 / s1 en RGB
          const mixWith = c.dark ? [26, 26, 25] : [252, 252, 251];
          const a = 0.10 + 0.55 * t;
          const col = base.map((v, i) => Math.round(v * a + mixWith[i] * (1 - a)));
          g.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
          g.fillRect(k * cell, r * cell, Math.ceil(cell) + 0.5, Math.ceil(cell) + 0.5);
        }
      }
      // Frontera de decisión (p = 0.5): marcha de cuadrados donde cambia el signo
      g.strokeStyle = c.dark ? 'rgba(255,255,255,.55)' : 'rgba(0,0,0,.45)';
      g.lineWidth = 1.4;
      g.beginPath();
      for (let r = 0; r < G; r++) for (let k = 0; k < G - 1; k++) {
        const a = grid[r * G + k] >= 0.5, b = grid[r * G + k + 1] >= 0.5;
        if (a !== b) { const x = (k + 1) * (W / G); g.moveTo(x, r * (W / G)); g.lineTo(x, (r + 1) * (W / G)); }
      }
      for (let r = 0; r < G - 1; r++) for (let k = 0; k < G; k++) {
        const a = grid[r * G + k] >= 0.5, b = grid[(r + 1) * G + k] >= 0.5;
        if (a !== b) { const yy = (r + 1) * (W / G); g.moveTo(k * (W / G), yy); g.lineTo((k + 1) * (W / G), yy); }
      }
      g.stroke();
    }

    if (showPoints) {
      for (let i = 0; i < n; i++) {
        const px = X[i * 2] * W, py = (1 - X[i * 2 + 1]) * Hh;
        const cls = y[i];
        const noisy = yTrue && yTrue[i] !== y[i];
        const wrong = pred && (pred[i] >= 0.5 ? 1 : 0) !== cls;
        g.beginPath();
        g.arc(px, py, i === sel ? 6 : 4, 0, Math.PI * 2);
        g.fillStyle = cls ? c.s2 : c.s1;
        g.fill();
        g.lineWidth = noisy ? 2 : 1;
        g.strokeStyle = noisy ? (c.dark ? '#fff' : '#000') : c.surface;
        g.stroke();
        if (wrong) {                       // aro alrededor de los mal clasificados
          g.beginPath(); g.arc(px, py, 7.5, 0, Math.PI * 2);
          g.strokeStyle = c.ink2; g.lineWidth = 1.2; g.stroke();
        }
      }
      if (sel >= 0 && sel < n) {
        g.beginPath(); g.arc(X[sel * 2] * W, (1 - X[sel * 2 + 1]) * Hh, 10, 0, Math.PI * 2);
        g.strokeStyle = c.ink; g.lineWidth = 2; g.stroke();
      }
    }
    g.strokeStyle = c.grid; g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, W - 1, Hh - 1);
  };

  /* ---------- 2. Curva de pérdida: entrenamiento vs validación por ronda ---------- */
  C.lossCurve = function (el, { hist, running, bestRound, metric = 'logloss' }) {
    const b = box(el, 0.54, 200, 300, { t: 18, r: 16, b: 40, l: 50 });
    if (!hist.length) { el.innerHTML = '<div class="empty">Pulsa <b>+10 rondas</b> para ver cómo evoluciona la pérdida.</div>'; return; }
    const isAcc = metric === 'acc';
    const tr = hist.map((h) => (isAcc ? h.trainAcc : h.train));
    const va = hist.map((h) => (isAcc ? h.val0Acc : h.val0));
    const all = tr.concat(va);
    const lo = Math.min(...all), hi = Math.max(...all);
    const pad = Math.max(0.01, (hi - lo) * 0.12);
    const n = hist.length;
    const x = lin(1, Math.max(n, 2), b.m.l, b.m.l + b.iw);
    const y = lin(Math.max(0, lo - pad), hi + pad, b.m.t + b.ih, b.m.t);
    const path = (arr) => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i + 1).toFixed(1)} ${y(v).toFixed(1)}`).join('');
    const best = bestRound && bestRound <= n
      ? `<line class="thr" x1="${x(bestRound)}" x2="${x(bestRound)}" y1="${b.m.t - 4}" y2="${b.m.t + b.ih}"/>` +
        `<text class="ink" x="${x(bestRound) + (x(bestRound) < b.m.l + b.iw - 90 ? 5 : -5)}" y="${b.m.t - 7}" text-anchor="${x(bestRound) < b.m.l + b.iw - 90 ? 'start' : 'end'}">mejor: ronda ${bestRound}</text>` : '';
    el.innerHTML = svgOpen(b, 'Pérdida de entrenamiento y validación por ronda de boosting') +
      gridY(b, y, niceTicks(Math.max(0, lo - pad), hi + pad, 4), (t) => t.toFixed(2)) +
      axisX(b, x, niceTicks(1, Math.max(n, 2), 5), (t) => Math.round(t)) + best +
      `<path d="${path(tr)}" fill="none" stroke-width="2.2" style="stroke:var(--s1)"/>` +
      `<path d="${path(va)}" fill="none" stroke-width="2.2" stroke-dasharray="5 3" style="stroke:var(--s2)"/>` +
      xTitle(b, 'ronda de boosting (nº de árboles)') + yTitle(b, isAcc ? 'exactitud' : 'logloss') +
      (running ? `<text class="mut" x="${b.m.l + b.iw}" y="${b.m.t + 12}" text-anchor="end">entrenando…</text>` : '') +
      `<g class="cursor" style="display:none"><line class="cur-line" y1="${b.m.t}" y2="${b.m.t + b.ih}"/></g>` +
      `<rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cl = el.querySelector('.cur-line');
    hover(el, (px) => {
      const i = clamp(Math.round(x.inv(px)) - 1, 0, n - 1), h = hist[i];
      cl.setAttribute('x1', x(i + 1)); cl.setAttribute('x2', x(i + 1));
      const gap = (isAcc ? h.trainAcc - h.val0Acc : h.val0 - h.train);
      return `<span class="tip-h">Ronda ${h.round}</span>` +
        row(dot('var(--s1)') + 'Entrenamiento', fmt(isAcc ? h.trainAcc : h.train, 3)) +
        row(dot('var(--s2)') + 'Validación', fmt(isAcc ? h.val0Acc : h.val0, 3)) +
        row('Brecha', fmt(gap, 3)) +
        '<span class="tip-sub">Si la validación sube mientras el entrenamiento baja, el modelo está sobreajustando.</span>';
    });
  };

  /* ---------- 3. Importancia por ganancia acumulada ---------- */
  C.importance = function (el, { gain, names }) {
    const d = gain.length, b = box(el, 0.42, 130, 200, { t: 16, r: 14, b: 34, l: 52 });
    const total = Array.from(gain).reduce((s, v) => s + v, 0) || 1;
    const mx = Math.max(...gain, 1e-9) * 1.15;
    const x = lin(0, mx, b.m.l, b.m.l + b.iw), slot = b.ih / d;
    let bars = '';
    for (let f = 0; f < d; f++) {
      const yy = b.m.t + f * slot + slot * 0.2, hgt = slot * 0.6;
      bars += `<rect x="${b.m.l}" y="${yy}" width="${Math.max(1, x(gain[f]) - b.m.l)}" height="${hgt}" rx="3" style="fill:var(--s1)"/>`;
      bars += `<text x="${b.m.l - 7}" y="${yy + hgt / 2 + 4}" text-anchor="end">${esc(names[f])}</text>`;
      bars += `<text class="mut" x="${x(gain[f]) + 6}" y="${yy + hgt / 2 + 4}">${(100 * gain[f] / total).toFixed(0)}%</text>`;
    }
    el.innerHTML = svgOpen(b, 'Importancia de cada variable por ganancia acumulada') +
      `<line class="axis" x1="${b.m.l}" x2="${b.m.l}" y1="${b.m.t}" y2="${b.m.t + b.ih}"/>` + bars +
      xTitle(b, 'ganancia total acumulada en los cortes') + '</svg>';
  };

  /* ---------- 4. Diagrama de calibración (fiabilidad) ---------- */
  C.calibration = function (el, { p, y, n, bins = 10 }) {
    const b = box(el, 0.92, 200, 280, { t: 14, r: 14, b: 38, l: 44 });
    const sum = new Float64Array(bins), cnt = new Float64Array(bins), pos = new Float64Array(bins);
    for (let i = 0; i < n; i++) {
      const k = clamp(Math.floor(p[i] * bins), 0, bins - 1);
      sum[k] += p[i]; cnt[k]++; pos[k] += y[i];
    }
    const x = lin(0, 1, b.m.l, b.m.l + b.iw), yy = lin(0, 1, b.m.t + b.ih, b.m.t);
    const pts = [];
    for (let k = 0; k < bins; k++) if (cnt[k] > 0) pts.push({ px: sum[k] / cnt[k], py: pos[k] / cnt[k], n: cnt[k], k });
    const line = pts.map((q, i) => `${i ? 'L' : 'M'}${x(q.px).toFixed(1)} ${yy(q.py).toFixed(1)}`).join('');
    const dots = pts.map((q) => `<circle cx="${x(q.px).toFixed(1)}" cy="${yy(q.py).toFixed(1)}" r="${clamp(2.5 + Math.sqrt(q.n) * 0.5, 3, 8).toFixed(1)}" style="fill:var(--s1);stroke:var(--surface);stroke-width:1.5"/>`).join('');
    const ticks = [0, 0.25, 0.5, 0.75, 1];
    el.innerHTML = svgOpen(b, 'Diagrama de calibración: probabilidad predicha frente a frecuencia observada') +
      gridY(b, yy, ticks, (t) => t.toFixed(2)) + axisX(b, x, ticks, (t) => t.toFixed(2)) +
      `<line class="ref" x1="${x(0)}" y1="${yy(0)}" x2="${x(1)}" y2="${yy(1)}"/>` +
      `<text class="mut" x="${x(0.62)}" y="${yy(0.62) + 15}">calibración perfecta</text>` +
      `<path d="${line}" fill="none" stroke-width="2" style="stroke:var(--s1)"/>` + dots +
      xTitle(b, 'probabilidad predicha') + yTitle(b, 'frecuencia real observada') +
      `<rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    hover(el, (px) => {
      const v = x.inv(px);
      let best = null, bd = Infinity;
      pts.forEach((q) => { const dd = Math.abs(q.px - v); if (dd < bd) { bd = dd; best = q; } });
      if (!best) return null;
      return `<span class="tip-h">Predicciones entre ${(best.k / bins).toFixed(1)} y ${((best.k + 1) / bins).toFixed(1)}</span>` +
        row('Probabilidad media predicha', fmt(best.px, 3)) + row('Frecuencia real', fmt(best.py, 3)) + row('Puntos', best.n) +
        '<span class="tip-sub">Sobre la diagonal = el modelo subestima; debajo = sobreestima.</span>';
    });
  };

  /* ---------- 5. Un árbol por dentro: los cortes de una ronda concreta ---------- */
  C.treeSplits = function (el, { tree, round }) {
    const b = box(el, 0.62, 170, 240, { t: 14, r: 12, b: 32, l: 36 });
    const x = lin(0, 1, b.m.l, b.m.l + b.iw), y = lin(0, 1, b.m.t + b.ih, b.m.t);
    // recorre el árbol acumulando el rectángulo de cada nodo
    const rects = [], lines = [];
    const stack = [[0, 0, 1, 0, 1]];
    while (stack.length) {
      const [node, x0, x1, y0, y1] = stack.pop();
      if (tree.feat[node] < 0) {
        const w = tree.weight[node];
        const t = clamp(Math.abs(w) / 1.2, 0, 1);
        rects.push(`<rect x="${x(x0)}" y="${y(y1)}" width="${Math.max(0, x(x1) - x(x0))}" height="${Math.max(0, y(y0) - y(y1))}" style="fill:${w >= 0 ? 'var(--s2)' : 'var(--s1)'};opacity:${(0.08 + t * 0.5).toFixed(2)}"/>`);
        continue;
      }
      const f = tree.feat[node], t = tree.thr[node];
      if (f === 0) {
        lines.push(`<line x1="${x(t)}" x2="${x(t)}" y1="${y(y1)}" y2="${y(y0)}" class="axis" stroke-width="1.5"/>`);
        stack.push([tree.left[node], x0, t, y0, y1], [tree.right[node], t, x1, y0, y1]);
      } else {
        lines.push(`<line x1="${x(x0)}" x2="${x(x1)}" y1="${y(t)}" y2="${y(t)}" class="axis" stroke-width="1.5"/>`);
        stack.push([tree.left[node], x0, x1, y0, t], [tree.right[node], x0, x1, t, y1]);
      }
    }
    el.innerHTML = svgOpen(b, `Cortes del árbol de la ronda ${round}`) +
      `<rect x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}" fill="none" class="axis"/>` +
      rects.join('') + lines.join('') +
      xTitle(b, `x₁ →   (árbol de la ronda ${round})`) + '</svg>';
  };
})();
