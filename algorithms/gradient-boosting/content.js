/* Contenido pedagógico del laboratorio de Gradient Boosting / XGBoost.
 * Las citas (Autor, año) enlazan con la bibliografía al final de la página. */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});
  const cite = (id, txt) => `<a class="cite" href="#ref-${id}">${txt}</a>`;
  ML.cite = cite;
  ML.MODEL_NAME = 'Gradient Boosting';

  /* Conceptos que comprueba tools/reading-check.js (definición antes del primer uso). */
  ML.READING_CONCEPTS = [
    ['boosting', 'boosting', 'ch1'], ['gradiente', 'gradiente', 'ch1'], ['hessiano', 'hessiano', 'ch1'], ['shrinkage', 'shrinkage', 'ch1'],
    ['hiperparámetro', 'hiperparámetro', 'top'], ['tocón', 'tocón', 'ch2'], ['sobreajuste', 'sobreajust', 'ch2'],
    ['early stopping', 'early stopping', 'ch2'], ['logloss', 'logloss', 'ch2'], ['poda', 'poda', 'ch2'],
    ['ROC-AUC', 'ROC-AUC', 'ch3'], ['AP', 'AP', 'ch3'], ['calibración', 'calibraci', 'ch3'],
    ['grid search', 'grid', 'ch4'], ['random search', 'random search', 'ch4'], ['bayesiana', 'bayesiana', 'ch4'],
  ];

  /* =====================================================================
   *  Hiperparámetros
   * ===================================================================== */
  ML.GB_PARAMS = [
    {
      id: 'nEstimators', code: 'n_estimators', symbol: 'T', title: 'Número de rondas (árboles)',
      short: 'Cuántos árboles se añaden, uno tras otro, cada uno corrigiendo el error del anterior.',
      what: 'Cada ronda ajusta un árbol pequeño al <span data-term="gradiente">gradiente</span> y al <span data-term="hessiano">hessiano</span> del error actual y lo <b>suma</b> (con peso <code>learning_rate</code>) a la predicción acumulada. A diferencia de un ensemble de <i>bagging</i> como Isolation Forest o Random Forest, los árboles <b>no son independientes</b>: cada uno depende de todos los anteriores.',
      controls: 'Controla cuánto se corrige el <b>sesgo</b> del modelo. Es la diferencia clave con <code>n_estimators</code> en bagging (que solo reduce varianza y nunca sobreajusta): aquí <b>más rondas sí pueden sobreajustar</b>, porque el modelo sigue corrigiendo hasta memorizar el ruido de entrenamiento.',
      upLabel: 'Si sigues sumando rondas', downLabel: 'Con pocas rondas',
      up: 'La pérdida de <b>entrenamiento</b> baja casi siempre (puede acercarse a 0). La de <b>validación</b> baja y en algún punto empieza a subir: ese mínimo es donde te gustaría haber parado (<span data-term="early">early stopping</span>).',
      down: 'El modelo apenas corrige la predicción inicial: <b>subajuste</b>. Con <code>learning_rate</code> alto hacen falta pocas rondas; con uno bajo, muchas más para llegar al mismo punto.',
      typical: 'No hay un valor universal: se decide <b>junto con learning_rate</b>, mirando dónde la pérdida de validación deja de bajar (capítulos 3 y 4). En la práctica se fija un T alto (500–1000) y se para con <i>early stopping</i> en vez de adivinar el número exacto de antemano.',
      tune: { level: 'high', label: 'Ajustar junto con learning_rate', text: 'No tiene un óptimo por separado: súbelo hasta el punto en que la validación empieza a empeorar, para el learning_rate que estés usando.' },
      real: '<b>Ejemplo:</b> un modelo de riesgo de crédito que se reentrena cada mes. Fijar T alto + <i>early stopping</i> evita tener que re-adivinar cuántas rondas hacen falta cada vez que cambian los datos.',
      watch: 'Mira la <b>curva de pérdida</b>: entrenamiento sigue bajando, validación forma una "U".',
      target: 'card-curve',
      live: (s) => `Con learning_rate = ${s.lr}, cada ronda mueve la predicción solo un <b>${Math.round(s.lr * 100)}%</b> de lo que "pediría" el gradiente completo: por eso hacen falta muchas rondas para que el efecto se acumule.`,
    },
    {
      id: 'lr', code: 'learning_rate', symbol: 'η', title: 'Tasa de aprendizaje (shrinkage)',
      short: 'Cuánto de la corrección de cada árbol se aplica de verdad. η = 1 aplica la corrección completa; η pequeño la frena.',
      what: 'Cada árbol nuevo se multiplica por η antes de sumarse: F(x) += η · árbol(x). Es el <span data-term="shrinkage">encogimiento</span> que propuso Friedman: en vez de un paso grande y arriesgado, muchos pasos pequeños y prudentes (' + cite('friedman2001', 'Friedman, 2001') + ').',
      controls: 'Es el intercambio central del boosting: η alto converge en pocas rondas pero sobreajusta antes y es más sensible al ruido de cada punto; η bajo generaliza mejor por ronda, pero necesita muchas más rondas para llegar igual de lejos.',
      upLabel: 'Con η grande (cerca de 1)', downLabel: 'Con η pequeño',
      up: 'Cada árbol corrige de más: el modelo se ajusta rápido a los datos de entrenamiento, <b>incluido su ruido</b>. La pérdida de validación forma una "U" pronunciada y temprana.',
      down: 'Cada corrección es tímida: hacen falta muchas más rondas para el mismo ajuste, pero el camino es más suave y con el mismo nº de rondas suele generalizar mejor.',
      typical: '0.05–0.3 es el rango habitual. Regla práctica: si divides η entre k, multiplica n_estimators por ~k para no quedarte corto.',
      tune: { level: 'high', label: 'Ajustar junto con n_estimators', text: 'Es la pareja que se busca junta: verás la cuadrícula completa en el capítulo 4.' },
      real: '<b>Ejemplo:</b> en competiciones con GBM se suele usar η muy bajo (0.01–0.05) con miles de rondas y <i>early stopping</i>, porque hay tiempo de cómputo de sobra y se quiere exprimir la última décima de AUC.',
      watch: 'El <b>barrido</b> de learning_rate y la cuadrícula learning_rate × n_estimators del capítulo 4.',
      target: 'card-sweep',
      live: (s) => `η = ${s.lr}: hacen falta del orden de <b>${Math.max(1, Math.round(1 / s.lr))} rondas</b> para que la suma de correcciones equivalga a "un solo árbol completo" (η = 1).`,
    },
    {
      id: 'maxDepth', code: 'max_depth', symbol: 'd', title: 'Profundidad máxima de cada árbol',
      short: 'Cuántos niveles de corte puede tener cada árbol. Controla qué interacciones puede captar UN árbol.',
      what: 'Un árbol de profundidad 1 (un <span data-term="tocon">tocón</span>) corta una sola vez: separa el plano en dos con una sola variable. Profundidad 2 permite combinar ambas variables — imprescindible en el <b>tablero (XOR)</b>, donde ninguna variable por separado dice nada.',
      controls: 'Controla la complejidad de <b>cada árbol</b>, no del ensemble completo. En boosting los árboles suelen mantenerse pequeños (2–6 niveles): la complejidad total sale de sumar <i>muchos</i> árboles simples, no de tener <i>pocos</i> árboles complejos como en Random Forest.',
      upLabel: 'Con profundidad alta', downLabel: 'Con profundidad 1 (tocones)',
      up: 'Cada árbol capta interacciones de mayor orden entre variables, pero también memoriza más ruido por árbol: con profundidad alta, pocas rondas ya sobreajustan.',
      down: 'El modelo es <b>aditivo puro</b> (suma efectos de cada variable por separado, sin interacción) y falla en el tablero sin importar cuántas rondas le des: no hay corte único que separe las clases.',
      typical: '3–6 en la mayoría de problemas tabulares; 1–2 cuando hay muchas variables ruidosas y se quiere limitar el sobreajuste de cada árbol.',
      tune: { level: 'high', label: 'Prioridad alta', text: 'Junto con learning_rate/n_estimators, el hiperparámetro que más cambia el resultado.' },
      real: '<b>Ejemplo:</b> en fraude, profundidad 4–6 permite reglas tipo "monto alto <b>y</b> hora inusual <b>y</b> país distinto al habitual" sin necesitar miles de rondas para aprenderlas.',
      watch: 'El escenario del <b>tablero</b>: profundidad 1 se queda en el azar, profundidad 2 lo resuelve.',
      target: 'card-map',
      live: (s) => `Con profundidad ${s.maxDepth}, un árbol tiene como máximo <b>${Math.pow(2, s.maxDepth)}</b> hojas (regiones distintas del plano).`,
    },
    {
      id: 'minChildWeight', code: 'min_child_weight', symbol: 'w', title: 'Peso mínimo por hoja',
      short: 'Suma mínima de hessiano (≈ cuántos puntos "inciertos" hacen falta) para permitir un corte. Evita hojas sostenidas por 1–2 puntos.',
      what: 'En pérdida logística, el hessiano de un punto es p(1−p) ≤ 0.25: los puntos donde el modelo ya está seguro pesan casi nada. <code>min_child_weight</code> exige que cada hoja acumule al menos ese peso, así que evita crear hojas hechas a la medida de un puñado de puntos ambiguos.',
      controls: 'Es un freno directo contra hojas demasiado específicas. Actúa junto con <code>max_depth</code>: con árboles profundos es más fácil terminar con hojas diminutas, así que subir este valor las evita.',
      upLabel: 'Con w alto', downLabel: 'Con w = 1 (por defecto)',
      up: 'Menos hojas y más grandes: cada una promedia más puntos. Más conservador; puede impedir aprender una región pequeña pero real.',
      down: 'Permite hojas hechas a la medida de 1–2 puntos: fácil de sobreajustar, sobre todo con profundidad alta.',
      typical: '1 por defecto; súbelo (5–20) si ves sobreajuste con árboles profundos o pocos datos por región.',
      tune: { level: 'med', label: 'Prioridad media', text: 'Ajústalo si gamma solo no controla el sobreajuste.' },
      real: '<b>Ejemplo:</b> con pocas ventas por tienda, un w alto evita que el árbol "invente" una regla para dos clientes atípicos de una tienda pequeña.',
      watch: 'Compara el <b>mapa de decisión</b> con w = 1 y w = 20 en el escenario de sobreajuste.',
      target: 'card-map',
      live: (s) => `Se necesitan del orden de <b>${Math.max(1, Math.round(s.minChildWeight / 0.25))} puntos</b> con el modelo "inseguro" (p ≈ 0.5) para sostener una hoja.`,
    },
    {
      id: 'gamma', code: 'gamma', symbol: 'γ', title: 'Ganancia mínima para cortar',
      short: 'Umbral de ganancia por debajo del cual un corte no se hace, aunque exista. Poda el árbol antes de construirlo.',
      what: 'Cada corte candidato reduce la pérdida regularizada en una cantidad (la <b>ganancia</b>). Si la mejor ganancia posible en un nodo es menor que γ, ese nodo se queda como hoja: es una poda "hacia adelante", no hace falta construir el corte para luego deshacerlo.',
      controls: 'Con γ = 0 (por defecto) se acepta cualquier corte que mejore, por poco que sea. Subirlo exige que el corte valga claramente la pena, produciendo árboles más pequeños y conservadores.',
      upLabel: 'Con γ alto', downLabel: 'Con γ = 0',
      up: 'Árboles con menos nodos: más sesgo, menos varianza. Con γ muy alto, ningún corte lo supera y cada árbol se queda en una sola hoja: deja de aportar nada.',
      down: 'Se aceptan cortes con ganancia mínima, que en datos ruidosos suelen ser ruido ajustado como si fuera señal.',
      typical: '0 para empezar; sube a 0.1–1 si el modelo sobreajusta con profundidad y min_child_weight ya razonables.',
      tune: { level: 'med', label: 'Prioridad media', text: 'Una alternativa a bajar max_depth cuando quieres podar de forma más selectiva.' },
      real: '<b>Ejemplo:</b> en un dataset pequeño y ruidoso, subir γ evita que el árbol "explique" una coincidencia de 3 puntos con un corte nuevo.',
      watch: 'El escenario de <b>sobreajuste</b>: compara la curva de pérdida con γ = 0 y γ = 1.',
      target: 'card-curve',
      live: (s) => `γ = ${s.gamma}: un corte solo se acepta si reduce la pérdida regularizada en más de <b>${s.gamma}</b>.`,
    },
    {
      id: 'lambda', code: 'lambda', symbol: 'λ', title: 'Regularización L2 de las hojas',
      short: 'Encoge el peso de cada hoja hacia 0. Con pocos puntos en una hoja, el peso se amortigua en vez de ajustarse por completo a ellos.',
      what: 'El peso óptimo de una hoja es w = −G/(H+λ). Sin λ, w = −G/H: el ajuste exacto a esos puntos. Con λ > 0 el denominador crece y w se acerca a 0, <b>más cuanto menos peso H</b> (menos puntos, o más inciertos) tenga la hoja.',
      controls: 'Es la contraparte de γ: en vez de impedir el corte, lo deja pero modera su efecto. Afecta más a hojas con pocos puntos que a las grandes.',
      upLabel: 'Con λ alto', downLabel: 'Con λ pequeño',
      up: 'Todas las hojas se moderan hacia 0, sobre todo las de pocos puntos: menos varianza, algo de sesgo.',
      down: 'Las hojas se ajustan casi exactamente a sus puntos, justo lo que produce sobreajuste cuando hay pocas observaciones por hoja.',
      typical: '1 por defecto suele ser razonable; sube a 5–10 con datasets pequeños o ruidosos.',
      tune: { level: 'med', label: 'Prioridad media', text: 'Súbelo si el sobreajuste persiste tras ajustar profundidad y learning_rate.' },
      real: '<b>Ejemplo:</b> con una categoría rara en los datos (una tienda nueva con 3 ventas), λ alto evita que su hoja prediga con una confianza que esos 3 datos no sostienen.',
      watch: 'Compara los pesos de las hojas (curva de pérdida) con λ = 0.1 y λ = 20 en el escenario de sobreajuste.',
      target: 'card-curve',
      live: (s) => `Una hoja con G = −2, H = 4 tendría peso ${(2 / (4 + s.lambda)).toFixed(2)} con λ = ${s.lambda} (sería ${(2 / 4).toFixed(2)} sin regularizar).`,
    },
    {
      id: 'subsample', code: 'subsample', symbol: 's', title: 'Submuestreo de filas por ronda',
      short: 'Fracción de las observaciones usada para ajustar CADA árbol, elegida al azar en cada ronda.',
      what: 'Con s < 1, cada ronda ve una submuestra distinta de filas: es <i>stochastic gradient boosting</i> (' + cite('friedman2002', 'Friedman, 2002') + '). Añade aleatoriedad entre rondas, en un espíritu parecido al submuestreo ψ de Isolation Forest, pero aquí el objetivo es reducir varianza del ensemble, no resolver masking.',
      controls: 'Introduce diversidad entre árboles consecutivos: si todos vieran siempre los mismos datos, tenderían a repetir el mismo tipo de error en las zonas difíciles.',
      upLabel: 'Con s = 1 (todas las filas)', downLabel: 'Con s bajo',
      up: 'Cada árbol ve todos los datos: más determinista, algo más propenso a que todos los árboles se equivoquen igual en los mismos puntos difíciles.',
      down: 'Cada árbol ve menos datos y son más distintos entre sí: puede mejorar la generalización y acelera cada ronda, pero con s muy bajo cada árbol es más ruidoso.',
      typical: '0.5–0.8 combinado con learning_rate bajo suele mejorar sobre s = 1 en datos con ruido.',
      tune: { level: 'low', label: 'Prioridad media-baja', text: 'Ajústalo después de fijar learning_rate, max_depth y n_estimators.' },
      real: '<b>Ejemplo:</b> con datos de sensores muy correlacionados entre sí, s = 0.6 evita que todas las rondas se apoyen en las mismas observaciones redundantes.',
      watch: 'El <b>barrido</b> de subsample: la varianza entre semillas crece a medida que s baja.',
      target: 'card-sweep',
      live: (s) => `Cada árbol se entrena con ≈ <b>${Math.round(s.subsample * 100)}%</b> de las filas de entrenamiento, distintas en cada ronda.`,
    },
    {
      id: 'seed', code: 'random_state', symbol: '#', title: 'Semilla aleatoria',
      short: 'Fija el submuestreo de filas (subsample) y el desempate entre cortes igual de buenos. Nunca se optimiza.',
      what: 'Misma semilla + mismos datos + mismos hiperparámetros ⇒ <b>exactamente el mismo modelo</b>. Cada ronda usa una semilla derivada de (random_state, número de ronda).',
      controls: 'Controla la <b>reproducibilidad</b>, no la calidad. Si cambiarla mueve mucho las métricas, con subsample < 1 la varianza del muestreo es alta.',
      upLabel: 'Buscar la "mejor" semilla', downLabel: 'Buena práctica',
      up: 'No existe una "mejor" semilla: elegirla por la métrica es sobreajustar al azar.',
      down: 'Fijarla para reproducir y reportar media ± desviación sobre varias semillas.',
      typical: 'Cualquier entero fijo. Nunca se usa como hiperparámetro a optimizar.',
      tune: { level: 'no', label: 'Nunca optimizar', text: 'Se fija para reproducir; se varía solo para medir la varianza.' },
      real: '<b>Ejemplo:</b> auditar por qué a un cliente se le negó un crédito hace tres meses exige poder reconstruir exactamente ese modelo: sin semilla fija (y datos versionados) no se puede.',
      watch: 'El <b>barrido</b> de random_state: cada punto es una semilla distinta.',
      target: 'card-sweep',
      live: () => 'Con subsample = 1, la semilla casi no importa (solo desempata cortes iguales). Con subsample bajo, sí.',
    },
  ];

  /* Estándar compartido del capítulo "Optimizar" (ver assets/js/core/tuning-doc.js). */
  ML.TUNING_DOC = {
    notTune: [
      { code: 'random_state', kind: 'repro',
        why: 'Elegir la semilla que da la mejor métrica es <b>sobreajustar al azar</b>: esa ventaja no existe en datos nuevos.',
        instead: 'fíjala para reproducir y auditar, y <b>varíala</b> (3–5 valores) para reportar media ± desviación.' },
      { code: 'objective / eval_metric', kind: 'decide',
        why: 'Qué se minimiza (logloss, AUC…) es una <b>decisión del problema</b> (¿te importa el ranking o la probabilidad calibrada?), no algo que se optimice buscando el valor que dé mejor número.',
        instead: 'elígelo según cómo se va a <b>usar</b> la predicción: un umbral fijo pide buena calibración; una lista ordenada pide buen ranking (AUC/AP).' },
      { code: 'n_estimators (sin early stopping)', kind: 'fix',
        why: 'Buscarlo por fuerza bruta como un hiperparámetro más duplica el costo de cada combinación de learning_rate/max_depth que pruebes.',
        instead: 'fija un valor alto y usa <b>early stopping</b> sobre un conjunto de validación: cada entrenamiento encuentra su propio punto de parada.' },
    ],
  };

  /* =====================================================================
   *  Métricas (tiles)
   * ===================================================================== */
  ML.GB_METRICS = {
    trainLoss: { label: 'Logloss (train)', group: 'fit',
      tip: '<span class="tip-h">Logloss de entrenamiento</span>Pérdida logística sobre los datos con los que se entrenó. <b>Siempre</b> baja o se mantiene al sumar rondas: por sí sola no dice si el modelo generaliza.<span class="tip-sub">No necesita datos nuevos, pero tampoco sirve para elegir el modelo.</span>' },
    valLoss: { label: 'Logloss (validación)', group: 'fit',
      tip: '<span class="tip-h">Logloss de validación</span>Pérdida sobre datos que el modelo <b>no usó</b> para entrenar. Baja y luego sube si hay sobreajuste: es la señal que de verdad importa.<span class="tip-sub">Necesita un conjunto separado del de entrenamiento.</span>' },
    gap: { label: 'Brecha train–val', group: 'fit', dir: -1,
      tip: '<span class="tip-h">Brecha de sobreajuste</span>Diferencia entre el logloss de validación y el de entrenamiento. Cerca de 0 = el modelo generaliza igual de bien de lo que memoriza; grande = está memorizando.<span class="tip-sub">No necesita etiquetas nuevas, solo el conjunto de validación.</span>' },
    valAcc: { label: 'Exactitud (val)', group: 'dec',
      tip: '<span class="tip-h">Exactitud en validación</span>Fracción de puntos de validación bien clasificados con el umbral 0.5.<span class="tip-sub">Engañosa con clases desbalanceadas: mira precisión/recall.</span>' },
    auc: { label: 'ROC-AUC (val)', group: 'rank',
      tip: '<span class="tip-h">ROC-AUC</span>Probabilidad de que un positivo real reciba una probabilidad mayor que un negativo al azar. 0.5 = azar, 1.0 = ranking perfecto.<span class="tip-sub">No depende del umbral.</span>' },
    ap: { label: 'AP (val)', group: 'rank',
      tip: '<span class="tip-h">Average Precision</span>Área bajo la curva Precisión–Recall. Más exigente que ROC-AUC cuando la clase positiva es rara.<span class="tip-sub">Un modelo al azar obtiene AP ≈ proporción de positivos.</span>' },
  };

  /* =====================================================================
   *  Escenarios guiados
   * ===================================================================== */
  ML.GB_LESSONS = [
    {
      id: 'first', title: 'Primera ronda a la enésima: corrigiendo el error', tag: 'n_estimators',
      setup: { dataset: 'blobs', nEstimators: 0, lr: 0.3, maxDepth: 2, minChildWeight: 1, gamma: 0, lambda: 1, subsample: 1, seed: 1, noise: 0, minor: 0.5, n: 300 }, active: 'nEstimators', target: 'card-map',
      steps: ['Con <b>0 rondas</b> el modelo solo predice la proporción de cada clase: el mapa es un color plano.', 'Pulsa <b>+1 ronda</b> varias veces y mira cómo aparece la frontera.'],
      look: 'Cada árbol nuevo corrige el error del anterior en la dirección del gradiente: la frontera se va afinando ronda a ronda, no de golpe. Con pocas rondas ya se distingue la forma general; los detalles finos llegan después.',
    },
    {
      id: 'lr-tradeoff', title: 'learning_rate alto vs. bajo: el mismo destino, otro camino', tag: 'learning_rate',
      setup: { dataset: 'moons', nEstimators: 15, lr: 0.9, maxDepth: 2, minChildWeight: 1, gamma: 0, lambda: 1, subsample: 1, seed: 1, noise: 0.05, minor: 0.5, n: 300 }, active: 'lr', target: 'card-curve',
      steps: ['Con η = 0.9 y 15 rondas, mira la curva: converge rápido pero irregular.', 'Baja η a 0.05 y sube n_estimators a 150. Compara la curva.'],
      look: 'Con η alto la pérdida de validación baja rápido pero puede empezar a subir pronto (sobreajuste temprano). Con η bajo y muchas más rondas, el camino es más suave y suele llegar a un mínimo de validación mejor — a costa de más cómputo.',
    },
    {
      id: 'overfit', title: 'Sobreajuste: cuando entrenamiento y validación se separan', tag: 'concepto',
      setup: { dataset: 'blobs', nEstimators: 200, lr: 0.3, maxDepth: 6, minChildWeight: 1, gamma: 0, lambda: 0.1, subsample: 1, seed: 1, noise: 0.18, minor: 0.5, n: 150 }, active: 'nEstimators', target: 'card-curve',
      steps: ['Estos datos tienen <b>18% de las etiquetas cambiadas al azar</b> (ruido de etiqueta).', 'Entrena hasta 200 rondas y mira la curva de pérdida completa, no solo el último valor.'],
      look: 'La pérdida de <b>entrenamiento</b> sigue bajando casi hasta 0: el árbol, profundo y sin apenas regularización, memoriza incluso las etiquetas ruidosas. La de <b>validación</b> baja al principio y luego <b>sube</b>: ese mínimo, no la ronda 200, es donde te habría convenido parar. Es la firma clásica del sobreajuste.',
    },
    {
      id: 'xor-depth', title: 'El tablero: por qué la profundidad importa', tag: 'max_depth',
      setup: { dataset: 'xor', nEstimators: 40, lr: 0.3, maxDepth: 1, minChildWeight: 1, gamma: 0, lambda: 1, subsample: 1, seed: 1, noise: 0, minor: 0.5, n: 300 }, active: 'maxDepth', target: 'card-map',
      steps: ['Con <b>profundidad 1</b> (tocones) entrena 40 rondas. El mapa apenas mejora del azar.', 'Sube max_depth a 2 y vuelve a entrenar.'],
      look: 'Con tocones, cada árbol solo puede cortar en x1 <i>o</i> en x2, nunca combinarlas: en el tablero (la clase depende de ambas a la vez) eso no aporta nada, sin importar cuántas rondas sumes. Con profundidad 2, un solo árbol ya puede combinar las dos variables y el mapa se resuelve con muchas menos rondas.',
    },
    {
      id: 'gamma-prune', title: 'gamma poda antes de que el ruido se aprenda', tag: 'gamma',
      setup: { dataset: 'blobs', nEstimators: 150, lr: 0.3, maxDepth: 5, minChildWeight: 1, gamma: 0, lambda: 0.1, subsample: 1, seed: 1, noise: 0.15, minor: 0.5, n: 150 }, active: 'gamma', target: 'card-curve',
      steps: ['Con γ = 0 y datos ruidosos, ya viste que la validación empeora tras muchas rondas.', 'Sube γ a 0.5 y luego a 2, reentrenando cada vez.'],
      look: 'Con γ alto, muchos de los cortes que antes se aceptaban (ganancia pequeña, casi siempre ruido) ahora se rechazan: los árboles quedan más simples y la brecha entre entrenamiento y validación se reduce. Con γ demasiado alto, el modelo deja de cortar del todo y subajusta.',
    },
    {
      id: 'imbalanced', title: 'Clase rara: cuando el 95% de exactitud no dice nada', tag: 'evaluación',
      setup: { dataset: 'blobs', nEstimators: 60, lr: 0.3, maxDepth: 3, minChildWeight: 1, gamma: 0, lambda: 1, subsample: 1, seed: 1, noise: 0, minor: 0.06, n: 600 }, active: 'nEstimators', target: 'card-roc',
      steps: ['Aquí solo el 6% de los puntos son de la clase positiva (como el fraude).', 'Mira la <b>exactitud</b> frente al <b>AP</b> y la curva Precisión–Recall.'],
      look: 'Un modelo que predijera siempre "clase 0" ya tendría ≈ 94% de exactitud sin haber aprendido nada. El <b>AP</b> y la curva PR sí exigen encontrar la clase rara: con clases desbalanceadas, mira siempre esas dos antes que la exactitud (' + cite('saito2015gb', 'Saito & Rehmsmeier, 2015') + ').',
    },
    {
      id: 'subsample-var', title: 'subsample: aleatoriedad entre rondas', tag: 'subsample',
      setup: { dataset: 'moons', nEstimators: 80, lr: 0.2, maxDepth: 3, minChildWeight: 1, gamma: 0, lambda: 1, subsample: 1, seed: 1, noise: 0.05, minor: 0.5, n: 300 }, active: 'subsample', target: 'card-sweep',
      steps: ['Con subsample = 1, cambia la semilla varias veces y mira cuánto se mueve el AUC de validación.', 'Baja subsample a 0.5 y repite.'],
      look: 'Con subsample bajo, cada semilla ve un conjunto de filas distinto por ronda, así que la varianza entre semillas sube (mira el <b>barrido</b> de random_state). A cambio, cada árbol individual es más rápido de entrenar y el ensemble en su conjunto suele generalizar algo mejor con datos ruidosos.',
    },
  ];

  /* =====================================================================
   *  Casos de la vida real
   * ===================================================================== */
  ML.CASES = [
    {
      id: 'fraude', name: 'Fraude con historial etiquetado', kicker: 'Finanzas',
      ctx: 'A diferencia del capítulo de <a href="../isolation-forest/">Isolation Forest</a> (sin etiquetas), aquí el banco ya tiene <b>meses de fraudes confirmados</b> por el equipo de investigación: casos reales, con etiqueta.',
      goal: 'Puntuar cada transacción nueva con la probabilidad de ser fraude, usando lo aprendido de los casos ya confirmados.',
      why: 'Con etiquetas de calidad disponibles, un modelo supervisado como XGBoost normalmente supera a uno no supervisado: aprende directamente el patrón de fraude confirmado, no solo "lo raro".',
      keys: [
        ['subsample / colsample', 'Valores 0.6–0.8 son un punto de partida habitual en fraude: hay muchas variables correlacionadas (monto, hora, comercio, dispositivo…).'],
        ['max_depth', '4–6 para capturar combinaciones tipo "monto alto + hora inusual + país distinto".'],
        ['n_estimators + early stopping', 'Fija un valor alto y para con un conjunto de validación separado en el tiempo (los fraudes más recientes).'],
      ],
      measure: '<b>AP / PR-AUC</b> y Precisión@k (cuántas alertas puede revisar el equipo por día); el ROC-AUC es optimista con tan pocos positivos (' + cite('saito2015gb', 'Saito & Rehmsmeier, 2015') + ').',
      trap: 'Las etiquetas de fraude confirmado están <b>sesgadas hacia lo que el proceso anterior ya sabía detectar</b>: el modelo puede aprender a repetir los puntos ciegos del sistema viejo. Complementa con revisión manual de una muestra de lo NO marcado.',
      sim: { lesson: 'imbalanced', label: 'Simular clase rara' },
    },
    {
      id: 'credito', name: 'Scoring de riesgo crediticio', kicker: 'Finanzas',
      ctx: 'Historial de miles de préstamos con variables del solicitante (ingresos, historial, deuda…) y si terminó en impago. Es el caso de uso clásico de GBM en la industria financiera.',
      goal: 'Estimar la <b>probabilidad de impago</b> para decidir si aprobar un crédito y en qué condiciones.',
      why: 'Los datos son tabulares, con variables heterogéneas e interacciones no lineales; XGBoost y variantes dominan estas competencias de forma consistente (' + cite('chen2016', 'Chen & Guestrin, 2016') + ').',
      keys: [
        ['lambda / gamma', 'Súbelos: un modelo de crédito auditable no debería depender de hojas sostenidas por 2–3 solicitantes atípicos.'],
        ['learning_rate', 'Bajo (0.02–0.05) con n_estimators alto: se puede permitir el cómputo extra y así se gana estabilidad.'],
        ['random_state', 'Fijo y documentado: una decisión de crédito debe ser reproducible y auditable.'],
      ],
      measure: '<b>Calibración</b> (¿una probabilidad de 0.2 corresponde de verdad a ~20% de impagos?) además de AUC: aquí la probabilidad se usa directamente, no solo el orden.',
      trap: 'Optimizar solo el AUC puede dar un modelo bien <i>ordenado</i> pero mal <b>calibrado</b>: útil para rankear solicitantes, engañoso si se usa la probabilidad como tasa de interés.',
      sim: { lesson: 'overfit', label: 'Simular sobreajuste' },
    },
    {
      id: 'churn', name: 'Predicción de abandono (churn)', kicker: 'Negocio',
      ctx: 'Una suscripción mensual con datos de uso (frecuencia, soporte, facturación) y si el cliente canceló el mes siguiente. Las señales de abandono suelen ser combinaciones sutiles, no un único indicador.',
      goal: 'Priorizar a qué clientes contactar con una oferta de retención antes de que cancelen.',
      why: 'El equipo de retención solo puede contactar a un número limitado de clientes: es exactamente el mismo problema de "lista corta ordenada" que en fraude, con costos de error distintos.',
      keys: [
        ['max_depth', '3–4: suele bastar con interacciones de 2–3 variables (p. ej. "poco uso + ticket de soporte reciente").'],
        ['n_estimators / learning_rate', 'Ajustados con early stopping sobre el mes más reciente, para no reentrenar "a ciegas" cada vez.'],
      ],
      measure: 'Precisión@k según la capacidad del equipo de retención, y AP si el abandono es minoritario.',
      trap: 'El "efecto placebo" de la retención: si contactas a todos los de score alto y algunos se quedan de todos modos, es fácil sobreestimar el impacto de la campaña sin un grupo de control.',
      sim: { lesson: 'imbalanced', label: 'Simular clase rara' },
    },
    {
      id: 'ctr', name: 'Ranking de anuncios (CTR)', kicker: 'Ciencia de datos',
      ctx: 'Cientos de miles de impresiones de anuncios con si el usuario hizo clic o no. El objetivo no es una decisión binaria sino <b>ordenar</b> qué anuncio mostrar primero.',
      goal: 'Ordenar candidatos por probabilidad de clic para decidir qué mostrar.',
      why: 'GBM es uno de los modelos de referencia histórico para CTR en sistemas de anuncios a gran escala, junto con variantes lineales con features de árboles (' + cite('he2014', 'He et al., 2014') + ').',
      keys: [
        ['subsample', 'Con millones de impresiones, subsample < 1 acelera cada ronda sin perder señal relevante.'],
        ['max_depth', 'Moderado (4–8): demasiado profundo sobreajusta a combinaciones de features que no se repetirán.'],
      ],
      measure: '<b>ROC-AUC</b> tiene sentido aquí (el objetivo es ordenar, no decidir con un umbral fijo); revisa también la calibración si el score alimenta una subasta.',
      trap: 'El comportamiento de los usuarios <b>cambia con el tiempo</b> (estacionalidad, campañas nuevas): un modelo válido hoy se degrada; monitorea el logloss en producción, no solo en el entrenamiento original.',
      sim: { lesson: 'subsample-var', label: 'Simular subsample' },
    },
    {
      id: 'salud', name: 'Apoyo diagnóstico', kicker: 'Salud',
      ctx: 'Variables clínicas (edad, resultados de laboratorio, síntomas) para estimar la probabilidad de una condición, como apoyo a la decisión de un profesional — nunca como reemplazo.',
      goal: 'Priorizar casos para revisión clínica según probabilidad estimada.',
      why: 'Con suficientes casos etiquetados, GBM suele rendir bien en datos clínicos tabulares y produce probabilidades (no solo una etiqueta), útiles para graduar la urgencia.',
      keys: [
        ['min_child_weight / lambda', 'Súbelos: en salud, una hoja que "inventa" una regla a partir de 3 pacientes atípicos es inaceptable.'],
        ['max_depth', 'Bajo-moderado (2–4): árboles más simples son más fáciles de auditar caso por caso.'],
      ],
      measure: 'Recall a una <b>precisión mínima aceptable</b> (un falso negativo suele costar mucho más que un falso positivo) y calibración de la probabilidad.',
      trap: 'El modelo hereda los <b>sesgos</b> de los datos históricos (qué pacientes llegaron a hacerse ciertas pruebas). Es una herramienta de priorización, no un diagnóstico.',
      sim: { lesson: 'gamma-prune', label: 'Simular poda con gamma' },
    },
  ];

  /* =====================================================================
   *  Preguntas de comprobación
   * ===================================================================== */
  ML.CHECKS = {
    ch1: [
      { q: '¿En qué se diferencia boosting de un ensemble como Isolation Forest o Random Forest?', a: 'En bagging los árboles son <b>independientes</b> (se promedian). En boosting cada árbol se ajusta al <b>error del anterior</b>: son secuenciales, y el orden importa.' },
      { q: '¿Por qué se usan gradiente Y hessiano, y no solo el gradiente?', a: 'El hessiano indica <b>cuánta confianza</b> tiene esa corrección (curvatura de la pérdida): permite dar un peso óptimo a cada hoja y decidir cuánta evidencia hace falta para justificar un corte (min_child_weight).' },
    ],
    ch2: [
      { q: 'Subes n_estimators de 50 a 300 y el logloss de entrenamiento baja de 0.3 a 0.05, pero el de validación sube de 0.32 a 0.41. ¿Qué concluyes?', a: 'Que el modelo está <b>sobreajustando</b> a partir de algún punto intermedio: deberías haber parado antes (early stopping), o bajar learning_rate/profundidad, o subir la regularización (gamma, lambda).' },
      { q: 'Con profundidad 1, un dataset tipo XOR no mejora sin importar cuántas rondas entrenes. ¿Por qué?', a: 'Porque un tocón solo puede cortar por <b>una</b> variable a la vez, y la clase en XOR depende de la <b>combinación</b> de ambas. Sumar más tocones no crea la interacción que falta; hace falta profundidad ≥ 2.' },
    ],
    ch3: [
      { q: '¿Por qué la pérdida de entrenamiento sola no sirve para elegir un modelo?', a: 'Porque en boosting <b>siempre</b> se puede bajar más entrenando más rondas o árboles más profundos, incluso memorizando ruido. Solo la pérdida en datos <b>no usados para entrenar</b> (validación) revela si eso generaliza.' },
      { q: 'Un modelo tiene 96% de exactitud con 4% de positivos reales. ¿Es bueno?', a: 'No se puede saber sin más: predecir siempre "negativo" ya da 96% de exactitud. Hay que mirar <b>AP, precisión y recall</b> de la clase positiva.' },
    ],
    ch4: [
      { q: '¿Por qué learning_rate y n_estimators no se ajustan por separado?', a: 'Porque el efecto de uno depende del otro: un learning_rate bajo necesita muchas más rondas para llegar al mismo punto que uno alto. Buscarlos por separado puede hacerte concluir que "más rondas siempre ayuda" cuando en realidad depende de con qué learning_rate.' },
      { q: '¿Cuándo conviene usar early stopping en vez de fijar n_estimators de antemano?', a: 'Casi siempre: en vez de adivinar T, se fija un T alto y se para automáticamente cuando la pérdida de validación deja de mejorar durante varias rondas seguidas.' },
    ],
    ch5: [
      { q: 'En riesgo de crédito, ¿por qué no basta con optimizar el AUC?', a: 'El AUC solo mide qué tan bien <b>ordena</b> el modelo a los solicitantes, no si la probabilidad que produce está <b>calibrada</b>. Si esa probabilidad se usa para fijar una tasa de interés, la calibración importa tanto como el orden.' },
    ],
  };

  ML.TAKEAWAYS = {
    ch1: ['Boosting suma árboles <b>secuenciales</b>, cada uno corrigiendo el error del anterior — al revés que un ensemble de bagging.', 'Cada árbol se ajusta al <b>gradiente</b> (dirección del error) y al <b>hessiano</b> (confianza) del residuo actual.', '<code>learning_rate</code> encoge cada corrección: pasos pequeños y prudentes en vez de uno grande.'],
    ch2: ['A diferencia de un ensemble de bagging, aquí <b>más rondas sí pueden sobreajustar</b>.', '<code>max_depth</code> controla qué interacciones puede captar cada árbol; con tocones (profundidad 1) no hay interacción posible.', 'gamma, lambda y min_child_weight son tres formas distintas de frenar el sobreajuste: podar antes de cortar, encoger las hojas, o exigir evidencia suficiente.'],
    ch3: ['La pérdida de <b>entrenamiento</b> no sirve para elegir un modelo: siempre se puede bajar más.', 'La <b>brecha</b> entre pérdida de entrenamiento y de validación es la señal de sobreajuste.', 'Con clases desbalanceadas, la exactitud engaña: usa AP y la curva Precisión–Recall.'],
    ch4: ['<code>learning_rate</code> y <code>n_estimators</code> se buscan <b>juntos</b>, no por separado.', '<i>Early stopping</i> sobre un conjunto de validación reemplaza la necesidad de adivinar n_estimators de antemano.', 'gamma, lambda y min_child_weight se ajustan <b>después</b>, para pulir el sobreajuste que quede.'],
    ch5: ['Con etiquetas de calidad, un modelo supervisado como XGBoost aprovecha esa información mejor que uno no supervisado.', 'El indicador correcto depende del uso: ranking (AUC), lista corta (AP, Precisión@k) o decisión con probabilidad (calibración).', 'Cada dominio tiene su propia trampa: sesgo heredado, campañas sin grupo de control, deriva temporal.'],
  };

  /* =====================================================================
   *  Glosario
   * ===================================================================== */
  ML.GLOSSARY = {
    boosting: ['Boosting', 'Familia de ensembles <b>secuenciales</b>: cada modelo nuevo corrige el error de la suma de los anteriores, al contrario que bagging (modelos independientes que se promedian).'],
    gradiente: ['Gradiente', 'Dirección en la que hay que mover la predicción para reducir la pérdida. En pérdida logística, el gradiente de cada punto es (probabilidad predicha − etiqueta real).'],
    hessiano: ['Hessiano', 'Curvatura de la pérdida: indica cuánta <b>confianza</b> tiene la corrección sugerida por el gradiente. En pérdida logística es p·(1−p).'],
    shrinkage: ['Shrinkage (encogimiento)', 'Multiplicar cada árbol nuevo por <code>learning_rate</code> antes de sumarlo, para que cada corrección sea prudente en vez de completa.'],
    hiperparametro: ['Hiperparámetro', 'Ajuste del algoritmo que se elige antes de entrenar (p. ej. profundidad de los árboles). Se distingue de los parámetros, que el algoritmo aprende de los datos.'],
    tocon: ['Tocón (stump)', 'Árbol de profundidad 1: un único corte, sobre una sola variable. No puede representar interacciones entre variables.'],
    sobreajuste: ['Sobreajuste (overfitting)', 'Cuando el modelo se ajusta al ruido específico de los datos de entrenamiento y deja de generalizar: la pérdida de entrenamiento sigue bajando mientras la de validación sube.'],
    early: ['Early stopping', 'Detener el entrenamiento cuando la pérdida en un conjunto de validación deja de mejorar durante varias rondas, en vez de fijar n_estimators de antemano.'],
    logloss: ['Logloss (pérdida logística)', 'Mide qué tan bien calibradas están las probabilidades predichas: penaliza mucho estar muy seguro y equivocado.'],
    poda: ['Poda (pruning)', 'Evitar o deshacer cortes que no aportan suficiente ganancia. En XGBoost, gamma poda "hacia adelante": ni siquiera se hace el corte si no supera el umbral.'],
    calibracion: ['Calibración', 'Si una probabilidad predicha de 0.2 corresponde, en la práctica, a que ese suceso ocurra ≈20% de las veces. Distinto de tener buen ranking (AUC).'],
    interaccion: ['Interacción', 'Cuando el efecto de una variable sobre la predicción depende del valor de otra. Un tocón no puede representarlas; un árbol de profundidad ≥ 2 sí.'],
    regularizacion: ['Regularización', 'Cualquier mecanismo que limite la complejidad del modelo para reducir el sobreajuste: aquí, gamma, lambda y min_child_weight.'],
    ap: ['Average Precision (AP)', 'Área bajo la curva Precisión–Recall. Más exigente que ROC-AUC cuando la clase positiva es rara.'],
    aucroc: ['ROC-AUC', 'Probabilidad de que un positivo real reciba una probabilidad mayor que un negativo elegido al azar.'],
    grid: ['Grid search', 'Búsqueda exhaustiva en una malla de valores predefinidos. El costo crece exponencialmente con el nº de hiperparámetros.'],
    random: ['Random search', 'Búsqueda que muestrea configuraciones al azar del espacio. Eficiente cuando pocos hiperparámetros importan.'],
    bayes: ['Optimización bayesiana', 'Construye un modelo de cómo el hiperparámetro afecta la métrica y elige la siguiente configuración más prometedora.'],
    semilla: ['Semilla (random_state)', 'Número que fija el generador aleatorio: misma semilla ⇒ mismo modelo.'],
  };
})();
