/* Gráficas del laboratorio de VAE (SVG y canvas a mano, sin librerías). */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});
  const { clamp, fmt, esc } = ML;
  const C = ML.charts, H = C._h;
  const { lin, logs, niceTicks, box, svgOpen, gridY, axisX, xTitle, yTitle, hover, dot, row } = H;

  /* ---------- Rejilla de imágenes 8×8 en un canvas ---------- */
  /** Dibuja imágenes (Float64Array en [0,1]) en una rejilla; `pairs` = alterna original/reconstruida. */
  C.imageGrid = function (cv, { imgs, S, cols, gap = 3, labels, marks, invert }) {
    const c = ML.colors();
    const n = imgs.length, rows = Math.ceil(n / cols);
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const W = Math.max(120, cv.clientWidth);
    const cell = (W - gap * (cols - 1)) / cols;
    const labH = labels ? 15 : 0;
    const Hh = rows * cell + gap * (rows - 1) + rows * labH;
    cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr);
    cv.style.height = Hh + 'px';
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, Hh);
    const px = cell / S;
    imgs.forEach((img, k) => {
      const r = Math.floor(k / cols), col = k % cols;
      const x0 = col * (cell + gap), y0 = r * (cell + gap + labH) + labH;
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        let v = clamp(img[y * S + x], 0, 1);
        if (invert) v = 1 - v;
        const t = c.dark ? v : 1 - v;                      // claro: 0 = blanco; oscuro: 0 = negro
        const lum = Math.round(t * 255);
        g.fillStyle = `rgb(${lum},${lum},${lum})`;
        g.fillRect(x0 + x * px, y0 + y * px, Math.ceil(px) + 0.4, Math.ceil(px) + 0.4);
      }
      g.strokeStyle = (marks && marks[k]) || c.grid;
      g.lineWidth = marks && marks[k] ? 2 : 1;
      g.strokeRect(x0 + 0.5, y0 + 0.5, cell - 1, cell - 1);
      if (labels && labels[k]) {
        g.fillStyle = c.ink2; g.font = '10.5px system-ui, sans-serif'; g.textAlign = 'center';
        g.fillText(labels[k], x0 + cell / 2, y0 - 4);
      }
    });
  };

  /* ---------- Curva de entrenamiento: reconstrucción y KL ---------- */
  C.trainCurve = function (el, { hist, beta, running, warmup }) {
    const b = box(el, 0.52, 190, 280, { t: 16, r: 54, b: 38, l: 48 });
    if (!hist.length) { el.innerHTML = '<div class="empty">Pulsa <b>+20 épocas</b> para ver cómo bajan las dos partes de la pérdida.</div>'; return; }
    const n = hist.length;
    const recs = hist.map((h) => h.rec), kls = hist.map((h) => h.kl);
    const hi = Math.max(...recs, 1), hiK = Math.max(...kls, 0.5);
    const x = lin(1, Math.max(n, 2), b.m.l, b.m.l + b.iw);
    const yR = lin(0, hi * 1.05, b.m.t + b.ih, b.m.t), yK = lin(0, hiK * 1.15, b.m.t + b.ih, b.m.t);
    const path = (arr, sc) => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i + 1).toFixed(1)} ${sc(v).toFixed(1)}`).join('');
    const warm = warmup > 0 && warmup <= n ? `<line class="ref" x1="${x(warmup)}" x2="${x(warmup)}" y1="${b.m.t}" y2="${b.m.t + b.ih}"/><text class="mut" x="${x(warmup) + 4}" y="${b.m.t + 11}">fin del recocido</text>` : '';
    el.innerHTML = svgOpen(b, 'Curva de entrenamiento: reconstrucción y KL por época') +
      gridY(b, yR, niceTicks(0, hi * 1.05, 4), (t) => t.toFixed(0)) + axisX(b, x, niceTicks(1, Math.max(n, 2), 5), (t) => Math.round(t)) + warm +
      `<path d="${path(recs, yR)}" fill="none" stroke-width="2.2" style="stroke:var(--s1)"/>` +
      `<path d="${path(kls, yK)}" fill="none" stroke-width="2.2" stroke-dasharray="5 3" style="stroke:var(--s2)"/>` +
      niceTicks(0, hiK * 1.15, 3).map((t) => `<text x="${b.m.l + b.iw + 6}" y="${yK(t) + 4}" style="fill:var(--s2)">${t.toFixed(1)}</text>`).join('') +
      xTitle(b, 'época') + yTitle(b, 'reconstrucción (nats)') +
      `<text class="title" transform="translate(${b.w - 8} ${b.m.t + b.ih / 2}) rotate(-90)" text-anchor="middle" style="fill:var(--s2)">KL (nats)</text>` +
      (running ? `<text class="mut" x="${b.m.l + b.iw}" y="${b.m.t + 11}" text-anchor="end">entrenando…</text>` : '') +
      `<g class="cursor" style="display:none"><line class="cur-line" y1="${b.m.t}" y2="${b.m.t + b.ih}"/></g><rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cl = el.querySelector('.cur-line');
    hover(el, (px) => {
      const i = clamp(Math.round(x.inv(px)) - 1, 0, n - 1), h = hist[i];
      cl.setAttribute('x1', x(i + 1)); cl.setAttribute('x2', x(i + 1));
      return `<span class="tip-h">Época ${h.epoch}</span>${row(dot('var(--s1)') + 'Reconstrucción', fmt(h.rec, 2))}${row(dot('var(--s2)') + 'KL', fmt(h.kl, 2))}${row('β efectivo', fmt(h.beta, 2))}${row('Pérdida total', fmt(h.rec + h.beta * h.kl, 2))}<span class="tip-sub">La pérdida que se minimiza es reconstrucción + β·KL.</span>`;
    });
  };

  /* ---------- Espacio latente: dispersión coloreada por un factor real ---------- */
  C.latentScatter = function (el, { mu, L, dimX, dimY, color, colorLabel, y, sel, onPick }) {
    const b = box(el, 0.96, 230, 340, { t: 14, r: 14, b: 40, l: 44 });
    const n = mu.length / L;
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) { xs.push(mu[i * L + dimX]); ys.push(mu[i * L + dimY]); }
    // el encuadre incluye siempre el círculo 2σ del prior, para que la comparación sea visible
    const pad = 0.35;
    const xl = Math.min(...xs, -2.2) - pad, xh = Math.max(...xs, 2.2) + pad;
    const yl = Math.min(...ys, -2.2) - pad, yh = Math.max(...ys, 2.2) + pad;
    const sp = Math.max(xh - xl, yh - yl) / 2, cx = (xl + xh) / 2, cy = (yl + yh) / 2;
    const clipId = 'lc' + Math.random().toString(36).slice(2, 8);
    const X = lin(cx - sp, cx + sp, b.m.l, b.m.l + b.iw), Y = lin(cy - sp, cy + sp, b.m.t + b.ih, b.m.t);
    // círculos del prior N(0, I): 1σ y 2σ
    const clip = `<clipPath id="${clipId}"><rect x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></clipPath>`;
    const pri = [1, 2].map((r) => `<circle cx="${X(0)}" cy="${Y(0)}" r="${Math.abs(X(r) - X(0))}" fill="none" class="ref"/>`).join('') +
      `<text class="mut" x="${X(0) + Math.abs(X(2) - X(0)) - 6}" y="${Y(0) - 6}" text-anchor="end">prior 2σ</text>`;
    const col = (i) => {
      if (y && y[i]) return 'var(--s2)';
      if (!color) return 'var(--s1)';
      const t = clamp(color[i], 0, 1);
      return `color-mix(in srgb, var(--s1) ${Math.round(12 + t * 80)}%, var(--surface))`;
    };
    let pts = '';
    for (let i = 0; i < n; i++) pts += `<circle cx="${X(xs[i]).toFixed(1)}" cy="${Y(ys[i]).toFixed(1)}" r="${y && y[i] ? 4 : 3.2}" style="fill:${col(i)};stroke:var(--surface);stroke-width:0.8"/>`;
    if (sel >= 0 && sel < n) pts += `<circle cx="${X(xs[sel])}" cy="${Y(ys[sel])}" r="8" fill="none" style="stroke:var(--ink);stroke-width:2"/>`;
    const ticks = niceTicks(cx - sp, cx + sp, 4);
    el.innerHTML = svgOpen(b, 'Espacio latente: media de cada observación') +
      gridY(b, Y, ticks, (t) => t.toFixed(1)) + axisX(b, X, ticks, (t) => t.toFixed(1)) +
      `<line class="ref" x1="${X(0)}" x2="${X(0)}" y1="${b.m.t}" y2="${b.m.t + b.ih}"/><line class="ref" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${Y(0)}" y2="${Y(0)}"/>` +
      `<defs>${clip}</defs><g clip-path="url(#${clipId})">` + pri + pts + '</g>' + xTitle(b, `z${dimX + 1}`) + yTitle(b, `z${dimY + 1}`) +
      (colorLabel ? `<text class="mut" x="${b.m.l}" y="${b.m.t + 11}">color = ${esc(colorLabel)}</text>` : '') +
      `<rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    hover(el, (px, py) => {
      let best = -1, bd = 300;
      for (let i = 0; i < n; i++) { const d = (X(xs[i]) - px) ** 2 + (Y(ys[i]) - py) ** 2; if (d < bd) { bd = d; best = i; } }
      if (best < 0) return null;
      el._hit = best;
      return `<span class="tip-h">Observación #${best + 1}${y && y[best] ? ' · anómala' : ''}</span>${row(`z${dimX + 1}`, fmt(xs[best], 2))}${row(`z${dimY + 1}`, fmt(ys[best], 2))}${color ? row(esc(colorLabel), fmt(color[best], 2)) : ''}<span class="tip-sub">Clic para ver su reconstrucción</span>`;
    });
    if (onPick) el.querySelector('.hit').addEventListener('pointerdown', () => { if (el._hit >= 0) onPick(el._hit); });
  };

  /* ---------- KL por dimensión latente (¿cuáles están vivas?) ---------- */
  C.klBars = function (el, { klDim, muVar }) {
    const L = klDim.length, b = box(el, L > 6 ? 0.42 : 0.5, 160, 230, { t: 20, r: 12, b: 40, l: 44 });
    const mx = Math.max(0.05, ...klDim) * 1.15;
    const x = lin(0, L, b.m.l, b.m.l + b.iw), y = lin(0, mx, b.m.t + b.ih, b.m.t), sl = b.iw / L;
    let bars = '';
    for (let j = 0; j < L; j++) {
      const dead = klDim[j] <= 0.01;
      bars += `<rect x="${b.m.l + j * sl + sl * 0.18}" y="${y(klDim[j])}" width="${sl * 0.64}" height="${Math.max(1, b.m.t + b.ih - y(klDim[j]))}" rx="3" style="fill:${dead ? 'var(--muted)' : 'var(--s1)'};opacity:${dead ? 0.45 : 1}"/>`;
      bars += `<text x="${b.m.l + j * sl + sl / 2}" y="${b.m.t + b.ih + 15}" text-anchor="middle" style="fill:${dead ? 'var(--muted)' : 'var(--ink-2)'}">z${j + 1}</text>`;
    }
    const thr = `<line class="thr" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${y(0.01)}" y2="${y(0.01)}"/><text class="mut" x="${b.m.l + b.iw - 2}" y="${y(0.01) - 5}" text-anchor="end">0.01 nats = apagada</text>`;
    el.innerHTML = svgOpen(b, 'KL por dimensión latente') + gridY(b, y, niceTicks(0, mx, 3), (t) => t.toFixed(2)) +
      `<line class="axis" x1="${b.m.l}" x2="${b.m.l + b.iw}" y1="${b.m.t + b.ih}" y2="${b.m.t + b.ih}"/>` + bars + thr +
      yTitle(b, 'KL (nats)') +
      `<g class="cursor" style="display:none"><rect class="cur-band" y="${b.m.t}" width="${sl}" height="${b.ih}"/></g><rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cb = el.querySelector('.cur-band');
    hover(el, (px) => {
      const j = Math.floor((px - b.m.l) / sl);
      if (j < 0 || j >= L) return null;
      cb.setAttribute('x', b.m.l + j * sl);
      return `<span class="tip-h">Dimensión z${j + 1}</span>${row('KL', fmt(klDim[j], 3) + ' nats')}${row('Varianza de μ', fmt(muVar[j], 3))}<span class="tip-sub">${klDim[j] <= 0.01 ? 'Apagada: el modelo no la usa (colapso de esta dimensión).' : 'Activa: transporta información sobre la entrada.'}</span>`;
    });
  };

  /* ---------- Histograma del error de reconstrucción por clase ---------- */
  C.errHist = function (el, { err, y, thr, hasBoth }) {
    const b = box(el, 0.52, 175, 250, { t: 20, r: 12, b: 38, l: 44 });
    const lo = 0, hi = Math.max(...err) * 1.02;
    const nb = 24, bw = (hi - lo) / nb;
    const c0 = new Array(nb).fill(0), c1 = new Array(nb).fill(0);
    let t0 = 0, t1 = 0;
    err.forEach((v, i) => { const k = clamp(Math.floor((v - lo) / bw), 0, nb - 1); if (y && y[i]) { c1[k]++; t1++; } else { c0[k]++; t0++; } });
    const sh = (c, t) => (t ? c / t : 0);
    let ymax = 0; for (let k = 0; k < nb; k++) ymax = Math.max(ymax, sh(c0[k], t0), sh(c1[k], t1));
    const x = lin(lo, hi, b.m.l, b.m.l + b.iw), Y = lin(0, ymax * 1.1, b.m.t + b.ih, b.m.t), sl = b.iw / nb;
    let bars = '';
    for (let k = 0; k < nb; k++) {
      const x0 = b.m.l + k * sl + sl / 2, w = Math.max(2, sl * (hasBoth ? 0.42 : 0.8));
      const t0p = Y(sh(c0[k], t0)), t1p = Y(sh(c1[k], t1));
      if (hasBoth) {
        bars += `<rect x="${x0 - w - 0.5}" y="${t0p}" width="${w}" height="${b.m.t + b.ih - t0p}" rx="1.5" style="fill:var(--s1)"/>`;
        bars += `<rect x="${x0 + 0.5}" y="${t1p}" width="${w}" height="${b.m.t + b.ih - t1p}" rx="1.5" style="fill:var(--s2)"/>`;
      } else bars += `<rect x="${x0 - w / 2}" y="${t0p}" width="${w}" height="${b.m.t + b.ih - t0p}" rx="1.5" style="fill:var(--s1)"/>`;
    }
    const tl = thr != null && thr <= hi ? `<line class="thr" x1="${x(thr)}" x2="${x(thr)}" y1="${b.m.t - 6}" y2="${b.m.t + b.ih}"/><text class="ink" x="${x(thr) + 4}" y="${b.m.t - 9}">umbral</text>` : '';
    el.innerHTML = svgOpen(b, 'Error de reconstrucción por clase') + gridY(b, Y, niceTicks(0, ymax * 1.1, 3), (t) => Math.round(t * 100) + '%') +
      axisX(b, x, niceTicks(lo, hi, 5), (t) => t.toFixed(0)) + bars + tl +
      xTitle(b, 'error de reconstrucción (nats) → peor reconstruido') + yTitle(b, hasBoth ? '% de cada clase' : '% de datos') +
      `<g class="cursor" style="display:none"><rect class="cur-band" y="${b.m.t}" width="${sl}" height="${b.ih}"/></g><rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cb = el.querySelector('.cur-band');
    hover(el, (px) => {
      const k = Math.floor((px - b.m.l) / sl);
      if (k < 0 || k >= nb) return null;
      cb.setAttribute('x', b.m.l + k * sl);
      return `<span class="tip-h">Error ${(lo + k * bw).toFixed(0)}–${(lo + (k + 1) * bw).toFixed(0)} nats</span>` +
        row(dot('var(--s1)') + (hasBoth ? 'Normales' : 'Datos'), c0[k]) + (hasBoth ? row(dot('var(--s2)') + 'Anómalos', c1[k]) : '') +
        '<span class="tip-sub">Lo que el VAE no logra reconstruir es candidato a anomalía.</span>';
    });
  };

  /* ---------- Barrido genérico reutilizado (usa C.sweep de Isolation Forest) ---------- */
})();
