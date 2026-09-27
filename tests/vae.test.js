const test = require('node:test');
const assert = require('node:assert/strict');
const VAE = require('../algorithms/vae/vae.js');
const SH = require('../algorithms/vae/shapes.js');
const M = require('../assets/js/core/metrics.js');
const U = require('../assets/js/core/unsup.js');

const trainOn = (d, o = {}, ep = 50) => {
  const m = new VAE({ D: 64, latent: o.latent ?? 2, hidden: o.hidden ?? 24, seed: o.seed ?? 1, lr: o.lr ?? 0.01, beta: o.beta ?? 1 });
  for (let e = 0; e < ep; e++) m.trainEpoch(d.X, d.n, o.batch ?? 16, VAE.annealed(o.beta ?? 1, e + 1, o.warmup ?? 0));
  return m;
};

test('datos: imágenes en [0,1], factores en [0,1] y anomalías al final', () => {
  const d = SH.generate('dotsize', 100, 3, 0.1, 0.02);
  assert.equal(d.D, 64); assert.equal(d.nF, 3);
  for (let i = 0; i < d.X.length; i++) assert.ok(d.X[i] >= 0 && d.X[i] <= 1);
  for (let i = 0; i < d.F.length; i++) assert.ok(d.F[i] >= 0 && d.F[i] <= 1);
  assert.equal(Array.from(d.y).filter(Boolean).length, 10);
  assert.equal(d.y[d.n - 1], 1);
  assert.equal(d.y[0], 0);
});

test('gradientes analíticos == numéricos (diferencias finitas)', () => {
  const d = SH.generate('dot', 8, 1, 0, 0);
  const m = new VAE({ D: 64, latent: 2, hidden: 12, seed: 3, lr: 0, beta: 1 });
  m.nrm = () => 0.3;                                  // ruido fijo: pérdida determinista
  const idx = Int32Array.from({ length: 8 }, (_, i) => i);
  const loss = () => {
    let r = 0, k = 0;
    for (let i = 0; i < 8; i++) {
      const off = i * 64, { mu, logvar } = m.encode(d.X, off);
      const z = new Float64Array(2);
      for (let j = 0; j < 2; j++) z[j] = mu[j] + Math.exp(0.5 * logvar[j]) * 0.3;
      const { xhat } = m.decode(z, 0);
      for (let t = 0; t < 64; t++) { const xv = d.X[off + t], p = Math.min(1 - 1e-7, Math.max(1e-7, xhat[t])); r -= xv * Math.log(p) + (1 - xv) * Math.log(1 - p); }
      for (let j = 0; j < 2; j++) k += -0.5 * (1 + logvar[j] - mu[j] * mu[j] - Math.exp(logvar[j]));
    }
    return (r + k) / 8;
  };
  const layers = [m.eh, m.emu, m.elv, m.dh, m.dout];
  const save = layers.map((L) => Float64Array.from(L.W));
  m.trainBatch(d.X, idx, 0, 8, 1);                    // lr=0: acumula sin mover los pesos
  const gAna = [0, 17, 123, 400].map((i) => m.dout.mW[i] / (1 - 0.9));   // Adam: m = (1−β₁)·g en t=1
  layers.forEach((L, i) => L.W.set(save[i]));
  const h = 1e-5, W = m.dout.W;
  [0, 17, 123, 400].forEach((i, k) => {
    const o = W[i];
    W[i] = o + h; const a = loss(); W[i] = o - h; const b = loss(); W[i] = o;
    const gNum = (a - b) / (2 * h);
    assert.ok(Math.abs(gAna[k] - gNum) / (Math.abs(gNum) + 1e-6) < 1e-3, `w${i}: ${gAna[k]} vs ${gNum}`);
  });
});

test('entrenar baja la pérdida y produce reconstrucciones mejores que la media', () => {
  const d = SH.generate('dot', 200, 7, 0, 0.02);
  const m = new VAE({ D: 64, latent: 2, hidden: 24, seed: 1, lr: 0.01, beta: 1 });
  const before = m.evaluate(d.X, d.n).rec;
  for (let e = 0; e < 40; e++) m.trainEpoch(d.X, d.n, 16, 1);
  const after = m.evaluate(d.X, d.n).rec;
  assert.ok(after < before * 0.6, `${before} → ${after}`);
  assert.ok(m.hist.length === 40 && m.hist[39].rec < m.hist[0].rec);
});

test('determinismo: misma semilla → mismo resultado; otra semilla → distinto', () => {
  const d = SH.generate('dot', 120, 7, 0, 0.02);
  const a = trainOn(d, { seed: 5 }, 12).evaluate(d.X, d.n);
  const b = trainOn(d, { seed: 5 }, 12).evaluate(d.X, d.n);
  const c = trainOn(d, { seed: 6 }, 12).evaluate(d.X, d.n);
  assert.equal(a.rec, b.rec);
  assert.notEqual(a.rec, c.rec);
});

test('KL en forma cerrada: q = prior ⇒ KL = 0, y crece al alejarse', () => {
  const kl = (mu, lv) => -0.5 * (1 + lv - mu * mu - Math.exp(lv));
  assert.ok(Math.abs(kl(0, 0)) < 1e-12);
  assert.ok(kl(1, 0) > 0 && kl(0, 1) > 0 && kl(0, -1) > 0);
  assert.ok(kl(2, 0) > kl(1, 0));
});

test('β alto provoca colapso posterior y el recocido lo rescata', () => {
  const d = SH.generate('dot', 300, 7, 0, 0.02);
  const collapsed = trainOn(d, { beta: 6, warmup: 0 }, 50).evaluate(d.X, d.n);
  assert.ok(collapsed.kl < 0.05 && collapsed.active === 0, `kl=${collapsed.kl}`);
  const ok = trainOn(d, { beta: 1, warmup: 0 }, 50).evaluate(d.X, d.n);
  assert.ok(ok.kl > 0.5 && ok.active === 2, `kl=${ok.kl} act=${ok.active}`);
  assert.ok(ok.rec < collapsed.rec, 'el modelo colapsado reconstruye peor');
  // el rescate se mide por el KL recuperado: contar dimensiones con un umbral fijo es ruidoso en la frontera
  const rescued = trainOn(d, { beta: 4, warmup: 25 }, 60).evaluate(d.X, d.n);
  const notRescued = trainOn(d, { beta: 4, warmup: 0 }, 60).evaluate(d.X, d.n);
  assert.ok(rescued.kl > notRescued.kl * 3, `recocido kl=${rescued.kl} vs ${notRescued.kl}`);
  assert.ok(rescued.active >= notRescued.active);
});

test('el VAE apaga las dimensiones latentes que no necesita', () => {
  const d = SH.generate('dot', 300, 7, 0, 0.02);
  const ev = trainOn(d, { latent: 8 }, 60).evaluate(d.X, d.n);
  assert.ok(ev.active < 8 && ev.active >= 2, `activas=${ev.active}`);
});

test('con menos dimensiones que factores, algún factor queda sin codificar', () => {
  const d = SH.generate('dotsize', 300, 7, 0, 0.02);     // 3 factores
  const align = (L) => {
    const ev = trainOn(d, { latent: L }, 70).evaluate(d.X, d.n);
    return d.factors.map((_, f) => {
      const fv = Float64Array.from({ length: d.n }, (_, i) => d.F[i * d.nF + f]);
      let best = 0;
      for (let j = 0; j < L; j++) best = Math.max(best, Math.abs(U.spearman(fv, Float64Array.from({ length: d.n }, (_, i) => ev.mu[i * L + j]))));
      return best;
    });
  };
  const a2 = align(2), a3 = align(3);
  assert.ok(Math.min(...a2) < 0.35, `con L=2 todos los factores estarían codificados: ${a2}`);
  assert.ok(Math.min(...a3) > Math.min(...a2), `L=3 debería codificar mejor el factor perdido: ${a3} vs ${a2}`);
});

test('el error de reconstrucción detecta anomalías', () => {
  const d = SH.generate('dotsize', 350, 7, 0.06, 0.02);
  const m = trainOn(d, { latent: 3 }, 60);
  const s = Float64Array.from({ length: d.n }, (_, i) => m.reconError(d.X, i * 64));
  assert.ok(M.rocAuc(s, d.y) > 0.9, 'AUC=' + M.rocAuc(s, d.y));
});

test('meanPathAtCheckpoints no aplica aquí; evaluate es coherente con reconError', () => {
  const d = SH.generate('bar', 80, 7, 0, 0.02);
  const m = trainOn(d, {}, 15);
  const ev = m.evaluate(d.X, d.n);
  for (let i = 0; i < 5; i++) assert.ok(Math.abs(ev.recErr[i * 10] - m.reconError(d.X, i * 10 * 64)) < 1e-9);
});

test('annealed: sube linealmente y se satura en beta', () => {
  assert.equal(VAE.annealed(4, 1, 0), 4);
  assert.equal(VAE.annealed(4, 5, 10), 2);
  assert.equal(VAE.annealed(4, 10, 10), 4);
  assert.equal(VAE.annealed(4, 50, 10), 4);
});

test('la red lineal (hidden = 0) entrena y es peor que la no lineal', () => {
  const d = SH.generate('bar', 250, 7, 0, 0.02);
  const lin = trainOn(d, { hidden: 0 }, 50).evaluate(d.X, d.n);
  const nonlin = trainOn(d, { hidden: 32 }, 50).evaluate(d.X, d.n);
  assert.ok(Number.isFinite(lin.rec) && lin.rec > 0);
  assert.ok(nonlin.rec < lin.rec, `lineal ${lin.rec} vs no lineal ${nonlin.rec}`);
});
