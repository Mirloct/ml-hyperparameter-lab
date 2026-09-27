/* Capítulo 4 — cuadrícula β × latent_dim: demuestra que los dos hiperparámetros INTERACTÚAN. */
(function () {
  'use strict';
  const ML = window.MLLab;
  const { $, $$, fmt, clamp, esc } = ML;
  const VAE = ML.VAE, SH = ML.shapes, U = ML.unsup, H = ML.charts._h;
  if (!$('#hmChart')) return;

  const BETAS = [0, 0.5, 1, 2, 4, 8];
  const LATS = [1, 2, 3, 4, 8];
  const S = { ds: 'dotsize', ep: 50, metric: 'rec', cells: null, running: false, token: 0 };

  function color(v, lo, hi, invert) {
    const t = clamp((v - lo) / (hi - lo || 1), 0, 1), u = invert ? 1 - t : t;
    return `color-mix(in srgb, var(--s1) ${Math.round(8 + u * 82)}%, var(--surface))`;
  }

  function draw() {
    const el = $('#hmChart');
    if (!S.cells) { el.innerHTML = '<div class="empty">Pulsa «Calcular cuadrícula» para entrenar los 30 modelos.</div>'; return; }
    const b = H.box(el, 0.46, 220, 330, { t: 16, r: 14, b: 44, l: 66 });
    const nC = BETAS.length, nR = LATS.length;
    const cw = b.iw / nC, ch = b.ih / nR;
    const key = S.metric, invert = key === 'rec';   // en reconstrucción, menor es mejor
    const vals = [];
    S.cells.forEach((row) => row.forEach((c) => { if (c && Number.isFinite(c[key])) vals.push(c[key]); }));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    let cells = '', best = null;
    LATS.forEach((L, ri) => BETAS.forEach((be, ci) => {
      const c = S.cells[ri][ci];
      const x = b.m.l + ci * cw, y = b.m.t + ri * ch;
      if (!c) { cells += `<rect x="${x + 1}" y="${y + 1}" width="${cw - 2}" height="${ch - 2}" rx="3" style="fill:var(--surface-2)"/>`; return; }
      const v = c[key];
      cells += `<rect x="${x + 1}" y="${y + 1}" width="${cw - 2}" height="${ch - 2}" rx="3" style="fill:${color(v, lo, hi, invert)}"/>`;
      const t = clamp((v - lo) / (hi - lo || 1), 0, 1), dark = (invert ? 1 - t : t) > 0.55;
      const txt = key === 'active' ? String(Math.round(v)) : v.toFixed(key === 'rec' ? 1 : 2);
      cells += `<text x="${x + cw / 2}" y="${y + ch / 2 + 4}" text-anchor="middle" style="fill:${dark ? '#fff' : 'var(--ink-2)'};font-size:11px;font-weight:600">${txt}</text>`;
      if (c.collapsed) cells += `<text x="${x + cw - 5}" y="${y + 13}" text-anchor="end" style="fill:${dark ? '#fff' : 'var(--bad)'};font-size:10px">✕</text>`;
      const better = invert ? v < (best ? best.v : Infinity) : v > (best ? best.v : -Infinity);
      if (!c.collapsed && better) best = { v, ri, ci, L, be };
    }));
    if (best) {
      const x = b.m.l + best.ci * cw, y = b.m.t + best.ri * ch;
      cells += `<rect x="${x + 2}" y="${y + 2}" width="${cw - 4}" height="${ch - 4}" rx="3" fill="none" style="stroke:var(--s2);stroke-width:2.5"/>`;
    }
    const xl = BETAS.map((be, ci) => `<text x="${b.m.l + ci * cw + cw / 2}" y="${b.m.t + b.ih + 16}" text-anchor="middle">${be}</text>`).join('');
    const yl = LATS.map((L, ri) => `<text x="${b.m.l - 8}" y="${b.m.t + ri * ch + ch / 2 + 4}" text-anchor="end">L = ${L}</text>`).join('');
    el.innerHTML = H.svgOpen(b, 'Cuadrícula beta por dimensión latente') + cells + xl + yl +
      H.xTitle(b, 'β (peso del KL)  →') + H.yTitle(b, 'dimensión latente') +
      `<g class="cursor" style="display:none"><rect class="cur-band" width="${cw}" height="${ch}"/></g><rect class="hit" x="${b.m.l}" y="${b.m.t}" width="${b.iw}" height="${b.ih}"/></svg>`;
    const cb = el.querySelector('.cur-band');
    H.hover(el, (px, py) => {
      const ci = Math.floor((px - b.m.l) / cw), ri = Math.floor((py - b.m.t) / ch);
      if (ci < 0 || ri < 0 || ci >= nC || ri >= nR || !S.cells[ri][ci]) return null;
      cb.setAttribute('x', b.m.l + ci * cw); cb.setAttribute('y', b.m.t + ri * ch);
      const c = S.cells[ri][ci];
      return `<span class="tip-h">β = ${BETAS[ci]} · latent_dim = ${LATS[ri]}</span>` +
        H.row('Reconstrucción', fmt(c.rec, 2)) + H.row('KL', fmt(c.kl, 2)) + H.row('Dimensiones activas', `${Math.round(c.active)} / ${LATS[ri]}`) +
        (Number.isFinite(c.dis) ? H.row('Alineación con factores', fmt(c.dis, 2)) : '') +
        (c.collapsed ? '<span class="tip-sub">✕ Colapso posterior: el modelo no usa el latente.</span>' : '');
    });
    read(best);
  }

  function read(best) {
    if (!S.cells) { $('#hmRead').innerHTML = ''; return; }
    const d = SH.DEFS[S.ds];
    const col = (ci) => LATS.map((L, ri) => S.cells[ri][ci]).filter(Boolean);
    const collapsedBetas = BETAS.filter((be, ci) => col(ci).length && col(ci).every((c) => c.collapsed));
    const nF = SH.generate(S.ds, 4, 1, 0, 0).nF;
    const parts = [];
    if (best) {
      const warn = S.metric === 'rec' && best.be === 0
        ? ' <b>Ojo:</b> con la métrica «reconstrucción», β = 0 gana <i>siempre</i>, porque no paga ninguna cuota de KL… pero entonces no es un VAE: el latente queda con huecos y deja de servir para generar. Es el ejemplo perfecto de por qué <b>no se elige un hiperparámetro por una sola métrica</b>. Cambia la métrica a «alineación con factores» y mira dónde cae el máximo.'
        : '';
      parts.push(`<p><span class="r-h">Mejor casilla (según «${$('#hmMetric').selectedOptions[0].textContent.split(' (')[0]}»)</span><br>β = <b>${best.be}</b> con latent_dim = <b>${best.L}</b>. Los datos tienen <b>${nF}</b> factores reales.${warn}</p>`);
    }
    if (collapsedBetas.length) {
      parts.push(`<p><span class="r-h">La interacción, en una frase</span><br>Con β ≥ <b>${collapsedBetas[0]}</b> <b>todas</b> las filas colapsan: da igual que subas latent_dim de 1 a 8, porque β apaga las dimensiones igual. Por eso <b>no se pueden ajustar por separado</b>: el mejor latent_dim depende de β, y buscar uno con el otro fijo te lleva a conclusiones falsas.</p>`);
    } else {
      parts.push(`<p><span class="r-h">La interacción</span><br>En este rango ninguna columna colapsa del todo. Fíjate igualmente en que la mejor latent_dim <b>cambia según la columna</b>: esa dependencia es la razón para buscarlos juntos.</p>`);
    }
    const r0 = S.cells[0].filter(Boolean), sub = LATS[0] < nF;
    if (sub && r0.length) parts.push(`<p><span class="r-h">Fila superior</span><br>Con latent_dim = ${LATS[0]} y ${nF} factores reales, ninguna β lo arregla: el cuello de botella es demasiado estrecho. Un hiperparámetro no compensa el mal ajuste de otro.</p>`);
    $('#hmRead').innerHTML = parts.join('');
  }

  async function run() {
    if (S.running) return;
    S.running = true; const token = ++S.token;
    $('#hmRun').disabled = true;
    const data = SH.generate(S.ds, 300, 7, 0, 0.02);
    S.cells = LATS.map(() => BETAS.map(() => null));
    let done = 0; const total = LATS.length * BETAS.length;
    for (let ri = 0; ri < LATS.length; ri++) {
      for (let ci = 0; ci < BETAS.length; ci++) {
        const L = LATS[ri], be = BETAS[ci];
        const m = new VAE({ D: data.D, latent: L, hidden: 24, seed: 1, lr: 0.01, beta: be });
        for (let e = 0; e < S.ep; e++) m.trainEpoch(data.X, data.n, 16, be);
        const v = m.evaluate(data.X, data.n);
        let dis = NaN;
        if (data.nF) {
          let sum = 0;
          for (let f = 0; f < data.nF; f++) {
            const fv = Float64Array.from({ length: data.n }, (_, i) => data.F[i * data.nF + f]);
            let bst = 0;
            for (let j = 0; j < L; j++) bst = Math.max(bst, Math.abs(U.spearman(fv, Float64Array.from({ length: data.n }, (_, i) => v.mu[i * L + j]))));
            sum += bst;
          }
          dis = sum / data.nF;
        }
        S.cells[ri][ci] = { rec: v.rec, kl: v.kl, active: v.active, dis, collapsed: v.kl < 0.05 };
        done++;
        $('#hmState').textContent = `entrenando ${done} / ${total}…`;
        draw();
        await new Promise((r) => setTimeout(r, 0));
        if (token !== S.token) { S.running = false; return; }
      }
    }
    S.running = false; $('#hmRun').disabled = false;
    $('#hmState').textContent = `${total} modelos entrenados (${S.ep} épocas cada uno).`;
    draw();
  }

  function init() {
    const sel = $('#hmDs');
    sel.innerHTML = Object.entries(SH.DEFS).map(([id, d]) => `<option value="${id}">${esc(d.name)}</option>`).join('');
    sel.value = S.ds;
    sel.addEventListener('change', (e) => { S.ds = e.target.value; S.cells = null; draw(); $('#hmState').textContent = 'Pulsa «Calcular cuadrícula».'; });
    $('#hmEp').addEventListener('change', (e) => { S.ep = +e.target.value; });
    $('#hmMetric').addEventListener('change', (e) => { S.metric = e.target.value; draw(); });
    $('#hmRun').addEventListener('click', run);
    draw();
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
