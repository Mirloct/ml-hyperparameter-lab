/*
 * Conjuntos de datos sintéticos 2D en [0,1]² para clasificación binaria supervisada.
 * A diferencia de Isolation Forest y el VAE, aquí el modelo SÍ ve la etiqueta y (es supervisado):
 * la pregunta ya no es "qué tan raro es este punto" sino "a qué clase pertenece".
 * Cada generación produce un conjunto de ENTRENAMIENTO y uno de VALIDACIÓN de la misma distribución,
 * necesarios para el capítulo 3 (sobreajuste = separación entre pérdida de train y de validación).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./gbm.js'));
  else { root.MLLab = root.MLLab || {}; root.MLLab.gbData = factory(root.MLLab.GBM); }
})(typeof self !== 'undefined' ? self : this, function (GBM) {
  'use strict';

  const clamp01 = (v) => Math.min(0.99, Math.max(0.01, v));
  function makeGauss(rng) {
    return function () { const u = 1 - rng(), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  }

  const SHAPES = {
    blobs: {
      name: 'Dos nubes', order: 0,
      blurb: 'Dos grupos casi separables por una sola frontera recta: el caso fácil, ideal para ver qué hace cada hiperparámetro sin que la forma de los datos estorbe.',
      gen(rng, g) {
        return rng() < 0.5 ? [clamp01(0.30 + 0.11 * g()), clamp01(0.34 + 0.11 * g()), 0]
          : [clamp01(0.68 + 0.11 * g()), clamp01(0.64 + 0.11 * g()), 1];
      },
    },
    moons: {
      name: 'Dos lunas', order: 1,
      blurb: 'Frontera curva y no lineal: un único corte no la resuelve, pero varios cortes encadenados (más profundidad o más rondas) sí.',
      gen(rng, g) {
        const up = rng() < 0.5, a = Math.PI * rng();
        const x = up ? Math.cos(a) : 1 - Math.cos(a), y = up ? Math.sin(a) : 0.5 - Math.sin(a);
        return [clamp01(0.5 + (x - 0.5) * 0.42 + 0.03 * g()), clamp01(0.5 + (y - 0.25) * 0.42 + 0.03 * g()), up ? 0 : 1];
      },
    },
    circles: {
      name: 'Círculos concéntricos', order: 2,
      blurb: 'La clase depende de la distancia al centro, no de una dirección: hace falta más de un nivel de profundidad para "rodear" el círculo interior.',
      gen(rng, g) {
        const inner = rng() < 0.5, a = rng() * 2 * Math.PI, r = (inner ? 0.14 : 0.34) + 0.02 * g();
        return [clamp01(0.5 + r * Math.cos(a)), clamp01(0.5 + r * Math.sin(a)), inner ? 1 : 0];
      },
    },
    xor: {
      name: 'Tablero (XOR)', order: 3,
      blurb: 'La clase es el "o exclusivo" de estar a la derecha y estar arriba. Ninguna variable por separado dice nada: un árbol de profundidad 1 (un tocón) no hace mejor que el azar.',
      gen(rng, g) {
        const qx = rng() < 0.5 ? 0 : 1, qy = rng() < 0.5 ? 0 : 1;
        return [clamp01((qx ? 0.75 : 0.25) + 0.09 * g()), clamp01((qy ? 0.75 : 0.25) + 0.09 * g()), qx ^ qy];
      },
    },
  };

  /** Fisher–Yates in-place. */
  function shuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
    return arr;
  }

  function sample(shapeId, n, rng, labelNoise, minorFrac) {
    const g = makeGauss(rng), shape = SHAPES[shapeId];
    let pts = [];
    for (let i = 0; i < n; i++) pts.push(shape.gen(rng, g));
    if (minorFrac < 0.5) {
      const c0 = pts.filter((p) => p[2] === 0), c1 = shuffle(pts.filter((p) => p[2] === 1), rng);
      const keep1 = Math.max(1, Math.round((minorFrac / (1 - minorFrac)) * c0.length));
      pts = shuffle(c0.concat(c1.slice(0, keep1)), rng);
    } else {
      shuffle(pts, rng);
    }
    const n2 = pts.length;
    const X = new Float64Array(n2 * 2), y = new Uint8Array(n2), yTrue = new Uint8Array(n2);
    let flipped = 0;
    pts.forEach((p, i) => {
      X[i * 2] = p[0]; X[i * 2 + 1] = p[1]; yTrue[i] = p[2];
      const flip = labelNoise > 0 && rng() < labelNoise;
      y[i] = flip ? 1 - p[2] : p[2];
      if (flip) flipped++;
    });
    return { X, y, yTrue, n: n2, flipped };
  }

  /** @returns {{train, val, shape, nPos, nPosVal}} */
  function generate(shapeId, n, seed, labelNoise, minorFrac) {
    const rng = GBM.mulberry32((seed * 2654435761) >>> 0);
    const def = SHAPES[shapeId] ? shapeId : 'blobs';
    const train = sample(def, n, rng, labelNoise, minorFrac);
    const val = sample(def, Math.max(80, Math.round(n * 0.6)), rng, labelNoise, minorFrac);
    return {
      shape: def, train, val,
      nPos: train.y.reduce((s, v) => s + v, 0), nPosVal: val.y.reduce((s, v) => s + v, 0),
    };
  }

  return { SHAPES, generate };
});
