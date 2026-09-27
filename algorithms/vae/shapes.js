/*
 * Datos para el laboratorio de VAE: imágenes 8×8 (64 píxeles en [0,1]) generadas a partir de
 * FACTORES conocidos (posición, tamaño, ángulo…). Conocer los factores permite comprobar si el
 * espacio latente los descubre — la prueba visual del "desenredo" de Higgins et al. (2017).
 *
 * Cada conjunto devuelve:
 *   X       Float64Array(n·64) en fila-mayor
 *   F       Float64Array(n·nF) valores reales de los factores, normalizados a [0,1]
 *   labels  nombres de los factores
 *   y       Uint8Array(n) 1 = anomalía (solo en los conjuntos con anomalías)
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./vae.js'));
  else { root.MLLab = root.MLLab || {}; root.MLLab.shapes = factory(root.MLLab.VAE); }
})(typeof self !== 'undefined' ? self : this, function (VAE) {
  'use strict';
  const S = 8, D = S * S;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  /** Dibuja un disco suave (antialias por distancia) en el lienzo img. */
  function disc(img, cx, cy, r, val) {
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
      const a = clamp(r + 0.5 - d, 0, 1);
      const i = y * S + x;
      img[i] = Math.max(img[i], a * val);
    }
  }
  /** Dibuja una barra centrada de longitud len y grosor w con ángulo ang (radianes). */
  function bar(img, cx, cy, len, w, ang, val) {
    const ca = Math.cos(ang), sa = Math.sin(ang);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
      const u = dx * ca + dy * sa, v = -dx * sa + dy * ca;
      const a = clamp(Math.min(len / 2 + 0.5 - Math.abs(u), w / 2 + 0.5 - Math.abs(v)), 0, 1);
      const i = y * S + x;
      img[i] = Math.max(img[i], a * val);
    }
  }

  const DEFS = {
    dot: {
      name: 'Punto móvil (2 factores)',
      blurb: 'Un disco del mismo tamaño que solo cambia de posición: <b>x</b> e <b>y</b>. Dos factores reales, así que basta un espacio latente de 2 dimensiones.',
      factors: ['posición x', 'posición y'],
      gen(rng) {
        const fx = rng(), fy = rng(), img = new Float64Array(D);
        disc(img, 1.6 + fx * 4.8, 1.6 + fy * 4.8, 1.25, 1);
        return { img, f: [fx, fy] };
      },
      anomaly(rng) {  // dos discos donde siempre hubo uno
        const img = new Float64Array(D);
        disc(img, 1.6 + rng() * 4.8, 1.6 + rng() * 4.8, 1.25, 1);
        disc(img, 1.6 + rng() * 4.8, 1.6 + rng() * 4.8, 1.25, 1);
        return { img, f: [0.5, 0.5] };
      },
    },
    dotsize: {
      name: 'Punto: posición y tamaño (3 factores)',
      blurb: 'Un disco que cambia de <b>posición</b> y de <b>tamaño</b>. Tres factores reales: con 2 dimensiones latentes el modelo tiene que sacrificar algo.',
      factors: ['posición x', 'posición y', 'tamaño'],
      gen(rng) {
        const fx = rng(), fy = rng(), fs = rng(), img = new Float64Array(D);
        disc(img, 1.8 + fx * 4.4, 1.8 + fy * 4.4, 0.75 + fs * 1.2, 1);
        return { img, f: [fx, fy, fs] };
      },
      anomaly(rng) {
        const img = new Float64Array(D);
        bar(img, 4, 4, 6, 1.4, rng() * Math.PI, 1);   // una barra donde siempre hubo discos
        return { img, f: [0.5, 0.5, 0.5] };
      },
    },
    bar: {
      name: 'Barra que gira (2 factores)',
      blurb: 'Una barra centrada que cambia de <b>ángulo</b> y <b>longitud</b>. El ángulo es circular: un espacio latente plano no puede representarlo sin una costura.',
      factors: ['ángulo', 'longitud'],
      gen(rng) {
        const fa = rng(), fl = rng(), img = new Float64Array(D);
        bar(img, 4, 4, 3 + fl * 4, 1.5, fa * Math.PI, 1);
        return { img, f: [fa, fl] };
      },
      anomaly(rng) {
        const img = new Float64Array(D);
        bar(img, 4, 4, 3 + rng() * 4, 1.5, rng() * Math.PI, 1);
        bar(img, 4, 4, 3 + rng() * 4, 1.5, rng() * Math.PI, 1);  // cruz
        return { img, f: [0.5, 0.5] };
      },
    },
    digits: {
      name: 'Trazos tipo dígito (3 factores)',
      blurb: 'Un trazo vertical con un travesaño cuya <b>altura</b>, <b>ancho</b> e <b>inclinación</b> varían. Más parecido a datos reales: los factores interactúan.',
      factors: ['altura del travesaño', 'ancho', 'inclinación'],
      gen(rng) {
        const fh = rng(), fw = rng(), fk = rng(), img = new Float64Array(D);
        const sk = (fk - 0.5) * 1.6;
        bar(img, 4, 4, 6, 1.3, Math.PI / 2 + sk * 0.25, 1);
        bar(img, 4 + sk * (fh - 0.5) * 2, 2 + fh * 4, 2 + fw * 3.5, 1.2, 0, 1);
        return { img, f: [fh, fw, fk] };
      },
      anomaly(rng) {
        const img = new Float64Array(D);
        disc(img, 2 + rng() * 4, 2 + rng() * 4, 2.2, 1);   // mancha en lugar de trazo
        return { img, f: [0.5, 0.5, 0.5] };
      },
    },
  };

  /**
   * @param {string} id conjunto
   * @param {number} n  nº de muestras
   * @param {number} seed
   * @param {number} anomalyFrac fracción de anomalías (0 = ninguna)
   * @param {number} noise ruido por píxel (desviación) — simula sensores imperfectos
   */
  function generate(id, n, seed, anomalyFrac, noise) {
    const def = DEFS[id] || DEFS.dot;
    const rng = VAE.mulberry32((seed * 2654435761) >>> 0), nrm = VAE.makeNormal(rng);
    const nF = def.factors.length;
    const X = new Float64Array(n * D), F = new Float64Array(n * nF), y = new Uint8Array(n);
    const nA = Math.round(n * (anomalyFrac || 0));
    for (let i = 0; i < n; i++) {
      const isA = i >= n - nA;
      const { img, f } = isA ? def.anomaly(rng) : def.gen(rng);
      for (let d = 0; d < D; d++) X[i * D + d] = clamp(img[d] + (noise ? nrm() * noise : 0), 0, 1);
      for (let j = 0; j < nF; j++) F[i * nF + j] = f[j];
      y[i] = isA ? 1 : 0;
    }
    return { X, F, y, n, D, S, nF, factors: def.factors };
  }

  return { DEFS, generate, D, S };
});
