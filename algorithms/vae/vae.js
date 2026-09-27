/*
 * Autoencoder Variacional (VAE) — Kingma & Welling (2014) — implementación didáctica en JS puro.
 *
 * Arquitectura (MLP totalmente conectado):
 *   x (D) → [enc] h (H, tanh) → μ (L), logσ² (L)
 *   z = μ + σ·ε   (truco de reparametrización, ε ~ N(0, I))
 *   z (L) → [dec] h (H, tanh) → x̂ (D, sigmoide)
 *
 * Objetivo (se MINIMIZA el ELBO negativo, con el peso β de Higgins et al. 2017):
 *   L = E[ BCE(x, x̂) ]  +  β · KL( q(z|x) ‖ N(0, I) )
 *   KL = −½ Σ_j ( 1 + logσ²_j − μ_j² − σ²_j )          (forma cerrada, gaussianas diagonales)
 *
 * Los gradientes están derivados a mano (sin autograd) para que el código sea legible y rápido.
 * Optimizador: Adam. Todo en Float64Array.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.MLLab = root.MLLab || {}; root.MLLab.VAE = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------- PRNG determinista (mulberry32) + normal por Box-Muller ---------- */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeNormal(rng) {
    let spare = null;
    return function () {
      if (spare !== null) { const v = spare; spare = null; return v; }
      const u = 1 - rng(), v = rng(), r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
  }
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  /* ---------- Capa densa con Adam ---------- */
  function Dense(nIn, nOut, rng) {
    const s = Math.sqrt(2 / (nIn + nOut));            // inicialización de Glorot (normal)
    const nrm = makeNormal(rng);
    const W = new Float64Array(nIn * nOut);
    for (let i = 0; i < W.length; i++) W[i] = nrm() * s;
    return {
      nIn, nOut, W, b: new Float64Array(nOut),
      gW: new Float64Array(nIn * nOut), gb: new Float64Array(nOut),
      mW: new Float64Array(nIn * nOut), vW: new Float64Array(nIn * nOut),
      mb: new Float64Array(nOut), vb: new Float64Array(nOut),
    };
  }
  /** y = W·x + b (x en `xs` desde el offset `xo`) */
  function forward(L, xs, xo, y) {
    const { nIn, nOut, W, b } = L;
    for (let o = 0; o < nOut; o++) y[o] = b[o];
    for (let i = 0; i < nIn; i++) {
      const xv = xs[xo + i]; if (xv === 0) continue;
      const row = i * nOut;
      for (let o = 0; o < nOut; o++) y[o] += xv * W[row + o];
    }
  }
  /** Acumula gradientes de la capa y devuelve dL/dx en `gx` (si se pide). */
  function backward(L, xs, xo, gy, gx) {
    const { nIn, nOut, W, gW, gb } = L;
    for (let o = 0; o < nOut; o++) gb[o] += gy[o];
    for (let i = 0; i < nIn; i++) {
      const xv = xs[xo + i], row = i * nOut;
      let acc = 0;
      for (let o = 0; o < nOut; o++) {
        const g = gy[o];
        gW[row + o] += xv * g;
        if (gx) acc += W[row + o] * g;
      }
      if (gx) gx[i] = acc;
    }
  }
  function adam(L, lr, t, scale, wd) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    const step = (g, m, v, p, n) => {
      for (let i = 0; i < n; i++) {
        let gi = g[i] * scale;
        if (wd) gi += wd * p[i];                       // decaimiento de pesos (L2)
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
        g[i] = 0;
      }
    };
    step(L.gW, L.mW, L.vW, L.W, L.W.length);
    step(L.gb, L.mb, L.vb, L.b, L.b.length);
  }

  class VAE {
    /**
     * @param {object} o { D, latent, hidden, seed, lr, beta, weightDecay, dropout }
     *   D        dimensión de entrada
     *   latent   dimensión del espacio latente (L)
     *   hidden   nº de unidades ocultas por capa (H); 0 = sin capa oculta (lineal)
     */
    constructor(o) {
      const rng = mulberry32((o.seed ?? 0) * 2654435761 + 12345);
      this.rng = rng; this.nrm = makeNormal(rng);
      this.D = o.D; this.L = o.latent; this.H = o.hidden;
      this.lr = o.lr ?? 0.01; this.beta = o.beta ?? 1; this.wd = o.weightDecay ?? 0;
      this.t = 0; this.epoch = 0;
      const H = this.H, D = this.D, Lz = this.L;
      this.eh = H ? Dense(D, H, rng) : null;            // encoder: x → h
      this.emu = Dense(H || D, Lz, rng);                // h → μ
      this.elv = Dense(H || D, Lz, rng);                // h → logσ²
      this.dh = H ? Dense(Lz, H, rng) : null;           // decoder: z → h
      this.dout = Dense(H || Lz, D, rng);               // h → logits de x̂
      // buffers reutilizados (evita asignar memoria en cada muestra)
      this.bh = new Float64Array(H || D); this.bmu = new Float64Array(Lz); this.blv = new Float64Array(Lz);
      this.bz = new Float64Array(Lz); this.bdh = new Float64Array(H || Lz); this.blogit = new Float64Array(D);
      this.bxhat = new Float64Array(D);
      this.g1 = new Float64Array(Math.max(D, H || 1, Lz)); this.g2 = new Float64Array(Math.max(D, H || 1, Lz));
      this.g3 = new Float64Array(Math.max(D, H || 1, Lz));
      this.hist = [];                                    // curva de entrenamiento por época
    }

    /** Paso hacia adelante del encoder. Devuelve {mu, logvar} (buffers internos). */
    encode(X, off) {
      const { H, eh, emu, elv, bh, bmu, blv } = this;
      let src = X, so = off;
      if (H) { forward(eh, X, off, bh); for (let i = 0; i < H; i++) bh[i] = Math.tanh(bh[i]); src = bh; so = 0; }
      forward(emu, src, so, bmu);
      forward(elv, src, so, blv);
      for (let j = 0; j < this.L; j++) blv[j] = Math.max(-12, Math.min(12, blv[j]));  // estabilidad numérica
      return { mu: bmu, logvar: blv, h: src, ho: so };
    }
    /** Paso hacia adelante del decoder desde z → x̂ (probabilidades). */
    decode(z, zo) {
      const { H, dh, dout, bdh, blogit, bxhat, D } = this;
      let src = z, so = zo;
      if (H) { forward(dh, z, zo, bdh); for (let i = 0; i < H; i++) bdh[i] = Math.tanh(bdh[i]); src = bdh; so = 0; }
      forward(dout, src, so, blogit);
      for (let i = 0; i < D; i++) bxhat[i] = sigmoid(blogit[i]);
      return { xhat: bxhat, h: src, ho: so };
    }

    /**
     * Una pasada de entrenamiento sobre un lote. Acumula gradientes y aplica Adam.
     * @returns {{rec:number, kl:number, n:number}} sumas del lote
     */
    trainBatch(X, idx, from, to, betaEff) {
      const { D, L: Lz, H } = this;
      const n = to - from;
      let recSum = 0, klSum = 0;
      const gmu = this.g1, glv = this.g2, gtmp = this.g3;
      const gz = new Float64Array(Lz), ghid = new Float64Array(Math.max(H || D, Lz));
      const eps = new Float64Array(Lz), gx = new Float64Array(D);

      for (let k = from; k < to; k++) {
        const off = idx[k] * D;
        const { mu, logvar, h: eHid, ho: eHo } = this.encode(X, off);
        // reparametrización
        for (let j = 0; j < Lz; j++) { eps[j] = this.nrm(); this.bz[j] = mu[j] + Math.exp(0.5 * logvar[j]) * eps[j]; }
        const { xhat, h: dHid, ho: dHo } = this.decode(this.bz, 0);

        // --- pérdidas ---
        for (let i = 0; i < D; i++) {
          const xv = X[off + i], p = Math.min(1 - 1e-7, Math.max(1e-7, xhat[i]));
          recSum -= xv * Math.log(p) + (1 - xv) * Math.log(1 - p);       // entropía cruzada binaria
          gx[i] = xhat[i] - xv;                                          // dL/dlogit (sigmoide + BCE)
        }
        for (let j = 0; j < Lz; j++) klSum += -0.5 * (1 + logvar[j] - mu[j] * mu[j] - Math.exp(logvar[j]));

        // --- retropropagación del término de reconstrucción ---
        backward(this.dout, dHid, dHo, gx, H ? ghid : gz);
        if (H) {
          for (let i = 0; i < H; i++) gtmp[i] = ghid[i] * (1 - dHid[i] * dHid[i]);   // tanh'
          backward(this.dh, this.bz, 0, gtmp, gz);
        }
        // z = μ + exp(½logσ²)·ε
        for (let j = 0; j < Lz; j++) {
          gmu[j] = gz[j];
          glv[j] = gz[j] * 0.5 * Math.exp(0.5 * logvar[j]) * eps[j];
          // --- gradiente del término KL (peso β) ---
          gmu[j] += betaEff * mu[j];
          glv[j] += betaEff * 0.5 * (Math.exp(logvar[j]) - 1);
        }
        backward(this.emu, eHid, eHo, gmu, H ? ghid : null);
        if (H) { for (let i = 0; i < H; i++) gtmp[i] = ghid[i]; }
        backward(this.elv, eHid, eHo, glv, H ? ghid : null);
        if (H) {
          for (let i = 0; i < H; i++) gtmp[i] = (gtmp[i] + ghid[i]) * (1 - eHid[i] * eHid[i]);
          backward(this.eh, X, off, gtmp, null);
        }
      }
      // --- actualización Adam (gradiente promedio del lote) ---
      this.t++;
      const scale = 1 / n;
      [this.eh, this.emu, this.elv, this.dh, this.dout].forEach((L) => { if (L) adam(L, this.lr, this.t, scale, this.wd); });
      return { rec: recSum, kl: klSum, n };
    }

    /**
     * Entrena una época completa (barajando los índices) y registra la curva.
     * @param {number} betaEff peso efectivo del KL (permite recocido / annealing)
     */
    trainEpoch(X, N, batchSize, betaEff) {
      const idx = new Int32Array(N);
      for (let i = 0; i < N; i++) idx[i] = i;
      for (let i = N - 1; i > 0; i--) { const j = Math.floor(this.rng() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
      let rec = 0, kl = 0;
      for (let s = 0; s < N; s += batchSize) rec += (({ rec: r, kl: k }) => { kl += k; return r; })(this.trainBatch(X, idx, s, Math.min(N, s + batchSize), betaEff));
      this.epoch++;
      const e = { epoch: this.epoch, rec: rec / N, kl: kl / N, beta: betaEff, elbo: rec / N + kl / N };
      this.hist.push(e);
      return e;
    }

    /** Evalúa el conjunto: pérdidas, latentes, reconstrucciones y KL por dimensión latente. */
    evaluate(X, N) {
      const { D, L: Lz } = this;
      const mu = new Float64Array(N * Lz), lv = new Float64Array(N * Lz);
      const recErr = new Float64Array(N), klPer = new Float64Array(N);
      const klDim = new Float64Array(Lz), muMean = new Float64Array(Lz), muM2 = new Float64Array(Lz);
      const recon = new Float64Array(N * D);
      let recSum = 0, klSum = 0;
      for (let i = 0; i < N; i++) {
        const off = i * D;
        const { mu: m, logvar: l } = this.encode(X, off);
        for (let j = 0; j < Lz; j++) { mu[i * Lz + j] = m[j]; lv[i * Lz + j] = l[j]; }
        const { xhat } = this.decode(m, 0);                    // reconstrucción determinista (usa μ)
        let r = 0;
        for (let d = 0; d < D; d++) {
          recon[off + d] = xhat[d];
          const xv = X[off + d], p = Math.min(1 - 1e-7, Math.max(1e-7, xhat[d]));
          r -= xv * Math.log(p) + (1 - xv) * Math.log(1 - p);
        }
        let k = 0;
        for (let j = 0; j < Lz; j++) {
          const kj = -0.5 * (1 + l[j] - m[j] * m[j] - Math.exp(l[j]));
          k += kj; klDim[j] += kj / N;
        }
        recErr[i] = r; klPer[i] = k; recSum += r; klSum += k;
      }
      // varianza de μ por dimensión: una dimensión colapsada tiene μ casi constante
      for (let j = 0; j < Lz; j++) {
        let s = 0; for (let i = 0; i < N; i++) s += mu[i * Lz + j];
        muMean[j] = s / N;
        let q = 0; for (let i = 0; i < N; i++) { const d = mu[i * Lz + j] - muMean[j]; q += d * d; }
        muM2[j] = q / N;
      }
      let active = 0;
      for (let j = 0; j < Lz; j++) if (klDim[j] > 0.01) active++;     // criterio usual: KL por dim > 0.01 nats
      return { mu, lv, recErr, klPer, recon, klDim, muVar: muM2, active,
        rec: recSum / N, kl: klSum / N, elbo: (recSum + klSum) / N };
    }

    /** Decodifica un punto latente arbitrario (para el mapa generativo). */
    generate(z) { const { xhat } = this.decode(z, 0); return Float64Array.from(xhat); }

    /** Error de reconstrucción de una muestra (score de anomalía). */
    reconError(X, off) {
      const { mu } = this.encode(X, off);
      const { xhat } = this.decode(mu, 0);
      let r = 0;
      for (let d = 0; d < this.D; d++) {
        const xv = X[off + d], p = Math.min(1 - 1e-7, Math.max(1e-7, xhat[d]));
        r -= xv * Math.log(p) + (1 - xv) * Math.log(1 - p);
      }
      return r;
    }
    nParams() {
      return [this.eh, this.emu, this.elv, this.dh, this.dout].reduce((s, L) => s + (L ? L.W.length + L.b.length : 0), 0);
    }
  }

  VAE.mulberry32 = mulberry32;
  VAE.makeNormal = makeNormal;
  VAE.sigmoid = sigmoid;
  /** Peso efectivo del KL con recocido lineal (Bowman et al., 2016): 0 → β en `warmup` épocas. */
  VAE.annealed = (beta, epoch, warmup) => (warmup > 0 ? beta * Math.min(1, epoch / warmup) : beta);
  return VAE;
});
