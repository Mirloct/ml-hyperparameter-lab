/*
 * Estándar compartido para el capítulo "Optimizar" de CUALQUIER modelo del sitio.
 *
 * Cada laboratorio declara ML.TUNING_DOC = { notTune: [...], effects?, methods? } y estas funciones
 * lo dibujan siempre igual: mismo lenguaje, mismas categorías, misma estructura visual.
 *
 * Cada entrada de `notTune` es:
 *   { code: 'n_estimators',        // el hiperparámetro tal como se llama en la librería
 *     why: 'texto HTML',           // por qué NO es un hiperparámetro a optimizar
 *     instead: 'texto HTML',       // qué hacer en su lugar
 *     kind: 'fix' | 'decide' | 'repro' }
 *   kind:
 *     fix    → se FIJA en un valor suficientemente bueno (plateau, presupuesto de cómputo)
 *     decide → es una DECISIÓN de negocio/producto, no un ajuste de calidad del modelo
 *     repro  → existe para REPRODUCIBILIDAD; optimizarlo es sobreajustar al azar
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.MLLab = root.MLLab || {}; root.MLLab.tuningDoc = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const KINDS = {
    fix: { label: 'Fíjalo', tip: '<span class="tip-h">Fíjalo</span>Existe un valor suficientemente bueno a partir del cual subir más solo cuesta tiempo y memoria. Se elige una vez (en el <b>plateau</b>) y no se vuelve a tocar.' },
    decide: { label: 'Decídelo', tip: '<span class="tip-h">Decídelo</span>No es una cuestión de calidad del modelo sino de <b>negocio</b>: costo de los errores, capacidad de revisión, riesgo aceptable. Optimizarlo contra una métrica agregada es engañoso.' },
    repro: { label: 'Solo reproducibilidad', tip: '<span class="tip-h">Solo reproducibilidad</span>Sirve para que el resultado sea repetible. Buscar el valor que da la mejor métrica es <b>sobreajustar al azar</b>: no generaliza.' },
  };

  /** Genera el HTML del bloque "Qué NO optimizar" a partir de la declaración del modelo. */
  function notTuneHTML(doc, esc) {
    const items = (doc && doc.notTune) || [];
    return `<div class="nt-grid">${items.map((it) => `
      <article class="card nt-card nt-${it.kind}">
        <header><code>${esc(it.code)}</code><span class="nt-kind" data-tip="${esc(KINDS[it.kind].tip)}">${KINDS[it.kind].label}</span></header>
        <p class="nt-why">${it.why}</p>
        <p class="nt-instead"><b>En su lugar:</b> ${it.instead}</p>
      </article>`).join('')}</div>
    <p class="fine nt-foot">Esta distinción se mantiene igual en todos los modelos del sitio: <b>fíjalo</b> (hay un plateau), <b>decídelo</b> (es una decisión de negocio) o <b>solo reproducibilidad</b> (optimizarlo es sobreajustar al azar). Lo que cambia de un modelo a otro es <i>qué</i> hiperparámetro cae en cada casilla.</p>`;
  }

  /** Rellena el contenedor [data-not-tune] si existe en la página. */
  function render(doc, esc) {
    const host = document.querySelector('[data-not-tune]');
    if (host) host.innerHTML = notTuneHTML(doc, esc || ((s) => s));
  }

  return { KINDS, notTuneHTML, render };
});
