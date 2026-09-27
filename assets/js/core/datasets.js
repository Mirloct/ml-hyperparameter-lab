/*
 * Conjuntos de datos sintéticos 2D en [0,1]². Cada generador devuelve puntos y etiquetas reales
 * (y=1 anómalo). El modelo NUNCA ve las etiquetas: solo se usan para evaluar.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('../../../algorithms/isolation-forest/iforest.js'));
  else { root.MLLab = root.MLLab || {}; root.MLLab.datasets = factory(root.MLLab.IsolationForest); }
})(typeof self !== 'undefined' ? self : this, function (IF) {
  'use strict';

  const clamp01 = (v) => Math.min(0.985, Math.max(0.015, v));

  function makeGauss(rng) {
    return function () {
      const u = 1 - rng(), v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
  }

  /** Puntos uniformes en el cuadrado, lejos (≥ minDist) de todos los inliers dados. */
  function scatterOutliers(m, inl, rng, minDist, region) {
    const out = [];
    let guard = 0;
    while (out.length < m && guard++ < 200000) {
      const p = region ? region(rng) : [0.03 + rng() * 0.94, 0.03 + rng() * 0.94];
      let ok = true;
      for (let i = 0; i < inl.length; i++) {
        const dx = inl[i][0] - p[0], dy = inl[i][1] - p[1];
        if (dx * dx + dy * dy < minDist * minDist) { ok = false; break; }
      }
      if (ok) out.push(p);
    }
    return out;
  }

  function pack(inliers, outliers) {
    const n = inliers.length + outliers.length;
    const X = new Float64Array(n * 2), y = new Uint8Array(n);
    inliers.forEach((p, i) => { X[i * 2] = p[0]; X[i * 2 + 1] = p[1]; });
    outliers.forEach((p, i) => {
      const k = inliers.length + i;
      X[k * 2] = p[0]; X[k * 2 + 1] = p[1]; y[k] = 1;
    });
    return { X, y, n };
  }

  const DEFS = {
    blob: {
      name: 'Un cúmulo + ruido',
      blurb: 'El caso ideal: una nube gaussiana y anomalías dispersas a su alrededor.',
      gen(N, m, rng) {
        const g = makeGauss(rng), inl = [];
        for (let i = 0; i < N - m; i++) {
          const a = g(), b = g();
          inl.push([clamp01(0.5 + 0.085 * a + 0.03 * b), clamp01(0.5 + 0.085 * b + 0.03 * a)]);
        }
        const wide = (r) => { // anomalías 'difíciles': nube ancha que se solapa con el borde del cúmulo
          const a = g(), b = g();
          return [clamp01(0.5 + 0.2 * a), clamp01(0.5 + 0.2 * b)];
        };
        return pack(inl, scatterOutliers(m, inl, rng, 0.055, wide));
      },
    },
    twoblobs: {
      name: 'Dos cúmulos (densidades distintas)',
      blurb: 'Un cúmulo denso y otro difuso. Las anomalías cerca del cúmulo difuso son más difíciles de distinguir.',
      gen(N, m, rng) {
        const g = makeGauss(rng), inl = [];
        const nA = Math.round((N - m) * 0.55);
        for (let i = 0; i < N - m; i++) {
          if (i < nA) inl.push([clamp01(0.28 + 0.05 * g()), clamp01(0.32 + 0.05 * g())]);
          else inl.push([clamp01(0.68 + 0.10 * g()), clamp01(0.66 + 0.10 * g())]);
        }
        return pack(inl, scatterOutliers(m, inl, rng, 0.085));
      },
    },
    moons: {
      name: 'Dos lunas',
      blurb: 'Estructura no convexa. Los cortes son paralelos a los ejes, pero el bosque aún logra "rodear" la forma.',
      gen(N, m, rng) {
        const g = makeGauss(rng), inl = [];
        const half = Math.floor((N - m) / 2);
        for (let i = 0; i < N - m; i++) {
          const up = i < half;
          const a = Math.PI * rng();
          const x = up ? Math.cos(a) : 1 - Math.cos(a);
          const yv = up ? Math.sin(a) : 0.5 - Math.sin(a);
          inl.push([clamp01(0.5 + (x - 0.5) * 0.36 + 0.018 * g()), clamp01(0.5 + (yv - 0.25) * 0.36 + 0.018 * g())]);
        }
        return pack(inl, scatterOutliers(m, inl, rng, 0.075));
      },
    },
    ring: {
      name: 'Anillo con hueco',
      blurb: 'Debilidad conocida: puntos en el hueco central quedan "rodeados" y son difíciles de aislar con cortes ortogonales.',
      gen(N, m, rng) {
        const g = makeGauss(rng), inl = [];
        for (let i = 0; i < N - m; i++) {
          const a = rng() * 2 * Math.PI, r = 0.30 + 0.018 * g();
          inl.push([clamp01(0.5 + r * Math.cos(a)), clamp01(0.5 + r * Math.sin(a))]);
        }
        const hole = (r) => { // mitad de las anomalías dentro del hueco
          if (r() < 0.5) {
            const a = r() * 2 * Math.PI, rad = 0.19 * Math.sqrt(r());
            return [0.5 + rad * Math.cos(a), 0.5 + rad * Math.sin(a)];
          }
          return [0.03 + r() * 0.94, 0.03 + r() * 0.94];
        };
        return pack(inl, scatterOutliers(m, inl, rng, 0.07, hole));
      },
    },
    masking: {
      name: 'Anomalías agrupadas (masking)',
      blurb: 'Parte de las anomalías forman un mini-cúmulo. Con muestras grandes se "esconden" entre sí (masking).',
      gen(N, m, rng) {
        const g = makeGauss(rng), inl = [];
        for (let i = 0; i < N - m; i++) inl.push([clamp01(0.36 + 0.09 * g()), clamp01(0.52 + 0.09 * g())]);
        const nCl = Math.round(m * 0.6), out = [];
        for (let i = 0; i < nCl; i++) out.push([clamp01(0.82 + 0.022 * g()), clamp01(0.22 + 0.022 * g())]);
        return pack(inl, out.concat(scatterOutliers(m - nCl, inl, rng, 0.14)));
      },
    },
  };

  /** @returns {{X:Float64Array,y:Uint8Array,n:number}} */
  function generate(id, N, anomalyFrac, seed) {
    const def = DEFS[id] || DEFS.blob;
    const rng = IF.mulberry32((seed * 2654435761) >>> 0);
    const m = Math.max(1, Math.round(N * anomalyFrac));
    return def.gen(N, m, rng);
  }

  return { DEFS, generate };
});
