/* Capítulos 3 y 4 — arena de optimización de hiperparámetros, prueba de los indicadores sin etiquetas
 * y la figura "grid vs random". Usa el paisaje precalculado (landscape.js). */
(function () {
  'use strict';
  const ML = window.MLLab;
  const { $, $$, fmt, clamp, esc } = ML;
  const TU = ML.tuning, U = ML.unsup, Ch = ML.charts, IF = ML.IsolationForest, D = ML.datasets;
  const LS = ML.landscapes;
  if (!LS || !TU) return;

  const METHODS = [
    { id: 'grid', name: 'Grid search', short: 'Grid', color: 'var(--s1)', band: false },
    { id: 'random', name: 'Random search', short: 'Random', color: 'var(--s2)', band: true },
    { id: 'bayes', name: 'Bayesiana (GP-EI)', short: 'Bayes', color: 'var(--s3)', band: true },
    { id: 'halving', name: 'Successive Halving', short: 'Halving', color: 'var(--s4)', band: true },
  ];
  const OBJ_ORDER = ['ap', 'mv', 'jac', 'agree'];
  const OBJ_SHORT = { ap: 'AP (etiquetas)', mv: 'Mass-Volume', jac: 'Estabilidad', agree: 'Acuerdo kNN' };
  const cfgTxt = (c) => `ψ=${c.psi} · mf=${c.mf} · bs=${c.bs ? 'T' : 'F'}`;
  const S = { ds: 'masking', obj: 'ap', budget: 800, show: 'random', runSeed: 1, on: { grid: true, random: true, bayes: true, halving: true }, token: 0, sim: null };
  let sharedDs = S.ds;

  /* =====================================================================
   *  Dataset compartido entre capítulos 3 y 4
   * ===================================================================== */
  function dsOptions() { return Object.entries(D.DEFS).map(([id, d]) => `<option value="${id}">${d.name}</option>`).join(''); }
  function setDataset(id, from) {
    if (!LS[id]) return;
    sharedDs = id; S.ds = id;
    ['#taDs', '#pxDs'].forEach((s) => { const el = $(s); if (el && el.value !== id) el.value = id; });
    renderProxies();
    if ($('#tuningArena').dataset.ready) scheduleSim();
  }

  /* =====================================================================
   *  ¿Sirven los indicadores sin etiquetas?  (Capítulo 3)
   * ===================================================================== */
  function proxyData(L, kind) {
    const cfgs = TU.allConfigs();
    const pts = cfgs.map((c) => {
      const m = TU.measure(L, c, TU.FULL_T);
      const raw = kind === 'mv' ? m.mv : kind === 'jac' ? m.jac : m.agree;
      return { x: kind === 'mv' ? -m.mv : raw, raw, y: m.ap, psi: c.psi, mf: c.mf, bs: c.bs };
    });
    const rho = U.spearman(Float64Array.from(pts, (p) => p.x), Float64Array.from(pts, (p) => p.y));
    return { pts, rho };
  }
  function renderProxies() {
    if (!$('#pxMV')) return;
    const L = LS[sharedDs], d = TU.DEFAULT_CFG, dm = TU.measure(L, d, TU.FULL_T);
    const spec = [['mv', '#pxMV', 'Mass-Volume', '−MV (mayor = mejor)', -dm.mv, 'Mass-Volume'], ['jac', '#pxJac', 'Estabilidad Jaccard@25', 'Jaccard entre semillas', dm.jac, 'la estabilidad'], ['agree', '#pxAgree', 'Acuerdo con kNN', 'Jaccard@25 con kNN', dm.agree, 'el acuerdo con kNN']];
    const oracleAP = TU.reference(L).oracleAP, lines = [];
    spec.forEach(([k, sel, title, xl, dx, name]) => {
      const { pts, rho } = proxyData(L, k);
      const best = pts.reduce((a, p) => (p.x > a.x ? p : a), pts[0]);
      Ch.proxyScatter($(sel), { title, pts, rho, xLabel: xl, dflt: { x: dx, y: dm.ap }, bestAP: best.y, oracleAP });
      const regret = oracleAP - best.y;
      lines.push({ name, rho, ap: best.y, regret, cfg: best });
    });
    const bad = lines.filter((l) => l.regret > 0.05);
    $('#pxRead').innerHTML = `<p><span class="r-h">Lectura en ${esc(D.DEFS[sharedDs].name)}</span><br>El mejor AP posible es <b>${fmt(oracleAP, 3)}</b>. La configuración que <b>maximiza</b> cada indicador rinde: ` +
      lines.map((l) => `${l.name}: <b>${fmt(l.ap, 3)}</b> (ρ = ${fmt(l.rho, 2)})`).join(' · ') + `.</p><p>` +
      (bad.length ? `<b>${bad.map((l) => l.name).join(' y ')} ${bad.length > 1 ? 'eligen' : 'elige'} mal aquí</b>. Fíjate: a veces la <b>correlación global es alta pero el máximo del indicador cae justo en una configuración mala</b> (p. ej. una muy estable pero ciega). Y con un cúmulo denso de anomalías, cualquier criterio basado en densidad o consenso las toma por "normales".` : 'En este conjunto los indicadores eligen configuraciones cercanas al óptimo. Prueba <i>Anomalías agrupadas</i> o <i>Anillo con hueco</i>: ahí fallan.') + '</p>';
  }

  /* =====================================================================
   *  Arena de optimización  (Capítulo 4)
   * ===================================================================== */
  const scheduleSim = ML.debounce(() => runSim(), 120);
  function buildArena() {
    const host = $('#tuningArena');
    host.innerHTML = `
      <div class="ta-ctl card">
        <div class="ta-row">
          <label class="ta-f"><span>Datos (paisaje precalculado)</span><select id="taDs">${dsOptions()}</select></label>
          <div class="ta-f"><span data-tip="<span class='tip-h'>¿Qué mide el tuner?</span>Lo que el optimizador intenta maximizar. Con etiquetas puedes usar AP; sin ellas, un indicador sustituto (proxy).">Objetivo que optimiza el tuner</span>
            <div class="seg" id="taObj" role="group" aria-label="Objetivo">${OBJ_ORDER.map((o) => `<button type="button" data-o="${o}" aria-pressed="${o === S.obj}">${OBJ_SHORT[o]}</button>`).join('')}</div></div>
          <label class="ta-f"><span>Presupuesto: <b id="taBudgetVal"></b></span><input id="taBudget" type="range" min="400" max="4000" step="100" value="${S.budget}"></label>
        </div>
        <div class="ta-row"><div class="ta-methods" id="taMethods" role="group" aria-label="Métodos a comparar"></div></div>
      </div>
      <div class="ta-grid">
        <article class="card chart-card">
          <header><h4>Paisaje: ¿dónde está lo bueno?</h4><p>Cada casilla es una configuración, coloreada por su <b>AP real</b> (T = 300). <span class="lg-diamond"></span> mejor posible · recuadro discontinuo = valores por defecto · círculos numerados = lo que probó la búsqueda.</p></header>
          <div class="ta-show"><span>Ver una búsqueda de:</span><div class="seg" id="taShow" role="group"></div><button class="btn sm" type="button" id="taAgain">↻ Otra búsqueda</button></div>
          <div class="chart" id="taLand" data-chart-tip></div>
        </article>
        <article class="card chart-card">
          <header><h4>¿Quién encuentra lo bueno más barato?</h4><p>AP <b>real</b> de la configuración que cada método recomendaría, según cuánto cómputo gastó. Línea = mediana de 60 búsquedas; banda = rango intercuartil.</p></header>
          <div class="chart" id="taCurves" data-chart-tip></div>
        </article>
      </div>
      <div class="card ta-result"><div id="taTable"></div><div class="reading" id="taRead" aria-live="polite"></div></div>`;
    $('#taDs').value = S.ds;
    $('#taMethods').innerHTML = METHODS.map((m) => `<label class="ta-chip"><input type="checkbox" data-m="${m.id}" ${S.on[m.id] ? 'checked' : ''}><span class="sw" style="background:${m.color}"></span>${m.name}</label>`).join('');
    $('#taShow').innerHTML = METHODS.map((m) => `<button type="button" data-m="${m.id}" aria-pressed="${m.id === S.show}">${m.name.split(' ')[0]}</button>`).join('');
    $('#taDs').addEventListener('change', (e) => setDataset(e.target.value));
    $$('#taObj button').forEach((b) => b.addEventListener('click', () => { S.obj = b.dataset.o; $$('#taObj button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); scheduleSim(); }));
    $('#taBudget').addEventListener('input', (e) => { S.budget = +e.target.value; $('#taBudgetVal').textContent = budgetTxt(); scheduleSim(); });
    $$('#taMethods input').forEach((i) => i.addEventListener('change', () => { S.on[i.dataset.m] = i.checked; scheduleSim(); }));
    $$('#taShow button').forEach((b) => b.addEventListener('click', () => { S.show = b.dataset.m; $$('#taShow button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); renderLandscape(); renderTable(); }));
    $('#taAgain').addEventListener('click', () => { S.runSeed++; renderLandscape(); renderTable(); });
    $('#taBudgetVal').textContent = budgetTxt();
    host.dataset.ready = '1';
  }
  const budgetTxt = () => `${S.budget.toLocaleString('es')} árboles (≈ ${Math.floor(S.budget / TU.EVAL_T)} evaluaciones de 100)`;

  const rngFor = (m) => IF.mulberry32(S.runSeed * 7919 + m.charCodeAt(0) * 31 + S.budget);
  function sampleRun(method) {
    const L = LS[S.ds], f = TU.OBJECTIVES[S.obj].f, r = rngFor(method);
    return method === 'grid' ? TU.runGrid(L, f, S.budget) : method === 'random' ? TU.runRandom(L, f, S.budget, r) : method === 'bayes' ? TU.runBayes(L, f, S.budget, r) : TU.runHalving(L, f, S.budget, r);
  }

  async function runSim() {
    const token = ++S.token, L = LS[S.ds], R = 60;
    const grid = Array.from({ length: 24 }, (_, i) => Math.round(100 + ((S.budget - 100) * i) / 23));
    const out = {};
    for (const m of METHODS) {
      if (!S.on[m.id]) continue;
      out[m.id] = TU.simulate(L, m.id, S.obj, S.budget, R, IF.mulberry32(17 + m.id.length * 101), grid);
      await new Promise((r) => setTimeout(r, 0));
      if (token !== S.token) return;
    }
    S.sim = { out, grid, ref: TU.reference(L) };
    renderCurves(); renderLandscape(); renderTable();
  }

  function renderCurves() {
    if (!S.sim) return;
    const { out, grid, ref } = S.sim;
    const series = METHODS.filter((m) => out[m.id]).map((m) => ({ name: m.short, color: m.color, band: m.band, med: out[m.id].med, lo: out[m.id].lo, hi: out[m.id].hi }));
    Ch.searchCurves($('#taCurves'), { grid, series, oracle: ref.oracleAP, dflt: ref.defaultAP, budget: S.budget });
  }
  function renderLandscape() {
    if (!S.sim) return;
    const L = LS[S.ds], ref = S.sim.ref, run = sampleRun(S.show);
    const seen = new Set(), marks = [];
    run.evals.forEach((e) => { const id = TU.cfgId(e.cfg); if (!seen.has(id)) { seen.add(id); marks.push({ cfg: e.cfg }); } });
    const fin = run.traj[run.traj.length - 1].cfg, fi = marks.findIndex((m) => TU.cfgId(m.cfg) === TU.cfgId(fin));
    if (fi >= 0) marks[fi].final = true;
    let lo = Infinity, hi = -Infinity;
    const vals = {};
    for (const c of TU.allConfigs()) { const v = TU.measure(L, c, TU.FULL_T).ap; vals[TU.cfgId(c)] = v; if (v < lo) lo = v; if (v > hi) hi = v; }
    Ch.landscape($('#taLand'), { space: TU.SPACE, values: (psi, mf, bs) => vals[TU.cfgId({ psi, mf, bs })], marks, oracle: ref.oracle, dflt: TU.DEFAULT_CFG, lo, hi });
    S.lastRun = { run, fin };
  }
  function reachCost(id) {
    const o = S.sim.out[id], ref = S.sim.ref, grid = S.sim.grid;
    for (let i = 0; i < grid.length; i++) if (Number.isFinite(o.med[i]) && o.med[i] >= ref.oracleAP - 0.005) return grid[i];
    return null;
  }
  function renderTable() {
    if (!S.sim || !S.lastRun) return;
    const L = LS[S.ds], { out, ref } = S.sim, f = TU.OBJECTIVES[S.obj].f;
    const rows = METHODS.filter((m) => out[m.id]).map((m) => {
      const run = sampleRun(m.id), fin = run.traj[run.traj.length - 1], mm = TU.measure(L, fin.cfg, TU.FULL_T);
      const med = out[m.id].med[out[m.id].med.length - 1], reach = reachCost(m.id);
      return `<tr><td><span class="sw" style="background:${m.color}"></span>${m.name}</td><td>${fin.cost}</td><td>${cfgTxt(fin.cfg)}</td><td>${fmt(mm.ap, 3)}</td><td>${fmt(ref.oracleAP - mm.ap, 3)}</td><td><b>${fmt(med, 3)}</b></td><td>${reach == null ? '<span class="muted-n">no llega</span>' : '≈ ' + Math.round(reach)}</td></tr>`;
    }).join('');
    $('#taTable').innerHTML = `<table class="ta-table"><thead><tr><th>Método</th><th>Árboles gastados</th><th>Config. elegida (esta búsqueda)</th><th>AP real</th><th>Pérdida vs. mejor</th><th data-tip="Mediana del AP real de lo elegido en 60 búsquedas simuladas.">AP real (mediana de 60)</th><th data-tip="<span class='tip-h'>Costo para llegar</span>Árboles entrenados hasta que la mediana de lo elegido queda a menos de 0.005 del mejor posible. Es la métrica que distingue a los métodos.">Árboles para llegar a ≤ 0.005 del óptimo</th></tr></thead><tbody>${rows}
      <tr class="ta-ref"><td>Valores por defecto</td><td>0</td><td>${cfgTxt(TU.DEFAULT_CFG)}</td><td>${fmt(ref.defaultAP, 3)}</td><td>${fmt(ref.oracleAP - ref.defaultAP, 3)}</td><td>—</td><td>—</td></tr>
      <tr class="ta-ref"><td>Mejor posible (oráculo)</td><td>—</td><td>${cfgTxt(ref.oracle)}</td><td>${fmt(ref.oracleAP, 3)}</td><td>0.000</td><td>—</td><td>—</td></tr></tbody></table>`;
    // lectura automática
    const meds = METHODS.filter((m) => out[m.id]).map((m) => ({ m, v: out[m.id].med[out[m.id].med.length - 1] })).sort((a, b) => b.v - a.v);
    const rho = U.spearman(Float64Array.from(TU.allConfigs(), (c) => f(TU.measure(L, c, TU.FULL_T))), Float64Array.from(TU.allConfigs(), (c) => TU.measure(L, c, TU.FULL_T).ap));
    const gain = ref.oracleAP - ref.defaultAP;
    const parts = [];
    parts.push(`<p><span class="r-h">Vale la pena tunear</span><br>${gain > 0.05 ? `En <b>${esc(D.DEFS[S.ds].name)}</b> los valores por defecto rinden AP ${fmt(ref.defaultAP, 3)} y lo mejor posible es ${fmt(ref.oracleAP, 3)}: hay <b>+${fmt(gain, 3)}</b> por ganar.` : `Aquí los valores por defecto ya están casi en el óptimo (AP ${fmt(ref.defaultAP, 3)} vs ${fmt(ref.oracleAP, 3)}): <b>tunear apenas compensa</b>. No siempre vale la pena.`}</p>`);
    if (meds.length > 1) {
      const spread = meds[0].v - meds[meds.length - 1].v;
      const reaches = meds.map((o) => ({ m: o.m, r: reachCost(o.m.id) })).filter((o) => o.r != null).sort((a, b) => a.r - b.r);
      let t = spread < 0.005 ? `Con este presupuesto los métodos quedan <b>prácticamente empatados</b> en calidad final (diferencia ${fmt(spread, 3)}). ` : `<b>${meds[0].m.name}</b> obtiene la mejor mediana (${fmt(meds[0].v, 3)}) y <b>${meds[meds.length - 1].m.name}</b> la peor (${fmt(meds[meds.length - 1].v, 3)}). `;
      if (reaches.length > 1) t += `Lo que sí los distingue es <b>cuánto cuesta llegar</b>: ${reaches[0].m.name} llega con ≈ ${Math.round(reaches[0].r)} árboles y ${reaches[reaches.length - 1].m.name} con ≈ ${Math.round(reaches[reaches.length - 1].r)}.`;
      else if (!reaches.length) t += 'Ninguno llega a estar a menos de 0.005 del óptimo con este presupuesto' + (S.obj !== 'ap' ? ' (optimizan otra cosa que el AP).' : ': prueba con más presupuesto.');
      parts.push(`<p><span class="r-h">Con este presupuesto</span><br>${t}</p>`);
    }
    if (S.obj !== 'ap') {
      const cfgs = TU.allConfigs(), pb = cfgs.reduce((a, c) => (f(TU.measure(L, c, TU.FULL_T)) > f(TU.measure(L, a, TU.FULL_T)) ? c : a), cfgs[0]), pbAP = TU.measure(L, pb, TU.FULL_T).ap, regret = ref.oracleAP - pbAP;
      parts.push(`<p><span class="r-h">Ojo: objetivo sin etiquetas</span><br>El tuner no ve el AP; optimiza <b>${OBJ_SHORT[S.obj]}</b>. Ese indicador correlaciona ρ = ${fmt(rho, 2)} con el AP real, pero la configuración que lo <b>maximiza</b> (${cfgTxt(pb)}) rinde AP <b>${fmt(pbAP, 3)}</b> frente a ${fmt(ref.oracleAP, 3)} del óptimo: ${regret <= 0.02 ? 'aquí el máximo del indicador coincide con lo bueno.' : regret <= 0.05 ? 'pierdes algo.' : '<b>el tuner puede elegir con total confianza una configuración mala</b>, sin que ningún indicador se lo advierta.'}</p>`);
    }
    $('#taRead').innerHTML = parts.join('');
  }

  /* =====================================================================
   *  Figura: por qué random supera a grid cuando pocos hiperparámetros importan
   * ===================================================================== */
  function gvrFigure(seed) {
    const host = $('#gvrFig'); if (!host) return;
    const W = Math.max(280, host.clientWidth || 520), panelW = (W - 24) / 2, H = 190, m = { l: 8, r: 8, t: 26, b: 36 };
    const rng = IF.mulberry32(seed), iw = panelW - m.l - m.r, ih = H - m.t - m.b;
    const grid = [], rand = [];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) grid.push([(i + 0.5) / 3, (j + 0.5) / 3]);
    for (let i = 0; i < 9; i++) rand.push([rng(), rng()]);
    const panel = (pts, ox, title, id) => {
      const xs = new Set(pts.map((p) => p[0].toFixed(3))).size;
      const bump = `<rect x="${ox + m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="url(#g${id})"/>`;
      const dots = pts.map((p) => `<circle cx="${(ox + m.l + p[0] * iw).toFixed(1)}" cy="${(m.t + (1 - p[1]) * ih).toFixed(1)}" r="4.6" style="fill:var(--ink);stroke:var(--surface);stroke-width:1.5"/>`).join('');
      const proj = pts.map((p) => `<circle cx="${(ox + m.l + p[0] * iw).toFixed(1)}" cy="${m.t + ih + 14}" r="3.2" style="fill:var(--ink-2)"/>`).join('');
      return `${bump}<rect x="${ox + m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="none" style="stroke:var(--axis)"/>${dots}${proj}
        <text class="ink" x="${ox + m.l}" y="${m.t - 10}">${title}</text>
        <text class="mut" x="${ox + m.l + iw}" y="${m.t + ih + 30}" text-anchor="end">${xs} valores distintos del hiperparámetro importante</text>`;
    };
    const defs = ['gg', 'gr'].map((id) => `<linearGradient id="g${id}" x1="0" x2="1" y1="0" y2="0"><stop offset="0" style="stop-color:var(--surface);stop-opacity:1"/><stop offset="0.5" style="stop-color:var(--s1);stop-opacity:.35"/><stop offset="1" style="stop-color:var(--surface);stop-opacity:1"/></linearGradient>`).join('');
    host.innerHTML = `<svg class="ch" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Grid frente a random search con 9 evaluaciones">${`<defs>${defs}</defs>`}${panel(grid, 0, 'Grid: 9 evaluaciones', 'gg')}${panel(rand, panelW + 24, 'Random: 9 evaluaciones', 'gr')}</svg>`;
  }

  function init() {
    if ($('#pxDs')) {
      $('#pxDs').innerHTML = dsOptions(); $('#pxDs').value = sharedDs;
      $('#pxDs').addEventListener('change', (e) => setDataset(e.target.value));
      renderProxies();
    }
    if ($('#tuningArena')) { buildArena(); runSim(); }
    if ($('#gvrFig')) {
      let seed = 3; gvrFigure(seed);
      $('#gvrAgain').addEventListener('click', () => gvrFigure(++seed));
      let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { gvrFigure(seed); renderProxies(); if (S.sim) { renderCurves(); renderLandscape(); } }, 150); });
    }
    window.addEventListener('lab:update', (e) => { if (LS[e.detail.dataset] && e.detail.dataset !== sharedDs) setDataset(e.detail.dataset); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
