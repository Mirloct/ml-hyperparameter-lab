/* Contenido pedagógico del laboratorio de Isolation Forest.
 * Las citas (Autor, año) enlazan con la bibliografía al final de la página. */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});
  const cite = (id, txt) => `<a class="cite" href="#ref-${id}">${txt}</a>`;
  ML.cite = cite;

  /* =====================================================================
   *  Hiperparámetros
   * ===================================================================== */
  ML.IF_PARAMS = [
    {
      id: 'nEstimators', code: 'n_estimators', symbol: 'T', title: 'Número de árboles',
      short: 'Cuántos árboles de aislamiento independientes forman el bosque. Cada árbol "opina" una profundidad de aislamiento y el bosque promedia todas las opiniones.',
      what: 'El bosque es un <b>conjunto (ensemble)</b> de <i>T</i> árboles construidos con azar. Cada árbol es un estimador muy ruidoso de qué tan fácil es aislar un punto; el score final usa el <b>promedio</b> de las profundidades, <i>E[h(x)]</i>.',
      controls: 'Controla la <span data-term="ruido">varianza</span> del score, no su sesgo: el error del promedio decrece aproximadamente como <i>1/√T</i>. Si el modelo es sistemáticamente ciego a algo (p. ej. <code>max_features = 0.5</code> en el anillo), más árboles <b>no lo arreglan</b>.',
      upLabel: '¿Y si lo subo y lo subo?', downLabel: 'Si lo bajas demasiado',
      up: 'Los scores se estabilizan, pero con <b>rendimientos decrecientes</b>: como el ruido cae como 1/√T, pasar de 100 a 400 árboles solo lo reduce a la mitad y cuesta <b>4× más</b> tiempo y memoria. Más allá del <i>plateau</i> subirlo ya no mejora nada medible. No hay sobreajuste (los árboles son independientes), pero <b>tampoco hay ganancia</b>. En el paper original, las longitudes de camino "convergen bastante antes de 100 árboles" (' + cite('liu2008', 'Liu et al., 2008') + ').',
      down: 'Scores ruidosos: dos ejecuciones con distinta semilla dan listas de alertas distintas. Con muy pocos árboles un punto normal puede "tener suerte" y aislarse pronto (falsa alarma).',
      typical: '100 (scikit-learn y paper). Aumenta hasta que el AP y la estabilidad dejen de moverse (mira el <b>Barrido</b>) y luego <b>fíjalo: no es un hiperparámetro para optimizar</b>. El argumento formal se demostró para Random Forest supervisado (' + cite('probst2018', 'Probst & Boulesteix, 2018') + ') y aplica por analogía a ensembles aleatorizados. En 29 conjuntos de datos, el AUC de un Random Forest apenas mejoraba pasados ~128 árboles (' + cite('oshiro2012', 'Oshiro et al., 2012') + ', citado en ese trabajo).',
      tune: { level: 'no', label: 'No optimizar', text: 'Fíjalo en un valor suficientemente alto: es un compromiso costo/estabilidad, no una búsqueda de "mejor valor".' },
      real: '<b>Ejemplo ilustrativo:</b> un banco puntúa cada transacción en tiempo real y cada árbol añade latencia. Con 100–200 árboles la lista diaria de alertas ya casi no cambia entre semillas; usar 1 000 multiplica el costo por transacción sin beneficio medible.',
      watch: 'Mira <b>Convergencia</b> (las líneas se aplanan), el <b>Barrido</b> (la banda entre semillas se estrecha y se ve el plateau) y el costo en milisegundos de la tabla.',
      target: 'card-sweep',
      live: (s, f) => `Con T = ${f.nEstimators} el ruido del promedio es ≈ 1/√T = <b>${(1 / Math.sqrt(f.nEstimators)).toFixed(2)}</b> veces el de un solo árbol. Con T = ${f.nEstimators * 4} sería <b>${(1 / Math.sqrt(f.nEstimators * 4)).toFixed(2)}</b>: la mitad, a 4× el costo.`,
    },
    {
      id: 'maxSamples', code: 'max_samples', symbol: 'ψ', title: 'Tamaño de submuestra',
      short: 'Cuántos puntos (elegidos al azar) usa cada árbol para construirse. No usa todos los datos: submuestrear es parte esencial del algoritmo.',
      what: 'Cada árbol se entrena con una <b>submuestra aleatoria</b> de ψ puntos. La profundidad máxima se deriva de ψ: <code>⌈log₂ ψ⌉</code> (no hace falta profundizar más: las anomalías se aíslan antes). ψ también normaliza el score mediante <i>c(ψ)</i>.',
      controls: 'Controla la <b>resolución</b> de cada árbol y combate dos problemas: el <span data-term="swamping">swamping</span> (normales cercanos a anomalías se confunden con ellas) y el <span data-term="masking">masking</span> (anomalías agrupadas se "esconden" entre sí). Muestras pequeñas dispersan los grupos de anomalías y las hacen aislables (' + cite('liu2012', 'Liu et al., 2012') + ').',
      up: 'Árboles más profundos y con más detalle de la forma de los datos, pero más masking/swamping y más costo. Ganas resolución en estructuras complejas (p. ej. lunas).',
      down: 'Árboles más rápidos, diversos y resistentes al masking; pero con ψ muy pequeño (&lt; 32) los árboles son demasiado toscos y los scores se vuelven imprecisos.',
      typical: '256 (paper original y scikit-learn), que según los autores basta en una gran variedad de datos. <b>No hay valor universal</b>: con anomalías agrupadas conviene ψ menor (32–128); con estructuras complejas y sin masking, mayor.',
      tune: { level: 'high', label: 'Prioridad alta', text: 'Es, junto con <code>max_features</code>, el hiperparámetro que más cambia el resultado según los datos (mira la tabla de tunabilidad del capítulo 4).' },
      real: '<b>Ejemplo:</b> en un ataque de denegación de servicio, miles de eventos casi idénticos forman un cúmulo denso de "anomalías": con muestras grandes se enmascaran. En un conjunto sintético con dos cúmulos anómalos densos (Mulcross), el AUC del paper pasó de 0.67 usando todos los datos a 0.91 con ψ = 128 (' + cite('liu2012', 'Liu et al., 2012') + ').',
      watch: 'Mira el <b>Barrido</b> de max_samples, la profundidad máxima derivada abajo y, en la vista <i>Un árbol</i>, qué puntos entraron a la muestra.',
      target: 'card-sweep',
      live: (s, f) => `ψ efectivo = <b>${f.psi}</b> → profundidad máxima ⌈log₂ ${f.psi}⌉ = <b>${f.maxDepth}</b>; c(ψ) = <b>${f.cPsi.toFixed(2)}</b> (profundidad esperada de un punto "promedio").`,
    },
    {
      id: 'contamination', code: 'contamination', symbol: 'c', title: 'Contaminación esperada',
      short: 'Proporción de anomalías que crees que hay en los datos. Solo sirve para fijar el umbral de decisión; no cambia el score ni el bosque.',
      what: 'Es la <b>fracción esperada de anomalías</b>. Con <code>\'auto\'</code> el umbral es fijo (score = 0.5, <code>offset_ = −0.5</code> en scikit-learn). Con un número, el umbral es el cuantil de los scores de entrenamiento que deja marcado exactamente ese porcentaje de puntos.',
      controls: 'Controla <b>solo el umbral</b> de decisión (paso de "score continuo" a "normal / anómalo"). El bosque y el ranking son idénticos: por eso ROC-AUC y AP <b>no cambian</b>, pero Precisión, Recall y F1 sí.',
      up: 'Se marcan más puntos: sube el <b>recall</b> (detectas más anomalías reales) y baja la <b>precisión</b> (más falsas alarmas).',
      down: 'Se marcan menos puntos: sube la <b>precisión</b>, pero se escapan anomalías (menos recall).',
      typical: 'Casi nunca se conoce (' + cite('perini2023', 'Perini et al., 2023') + '). Úsalo con conocimiento del negocio o con la <b>capacidad de revisión</b> del equipo. Si no lo conoces, trabaja con el <b>ranking</b> (top-k) en lugar de fijar un porcentaje.',
      tune: { level: 'no', label: 'Es una decisión, no un ajuste', text: 'No lo "optimices" para F1 a ciegas: fija el umbral por costo de errores y capacidad de revisión. No afecta al bosque.' },
      real: '<b>Ejemplo:</b> en el conjunto público de fraude de tarjetas de la ULB solo 492 de 284 807 transacciones (0.172 %) son fraude. Si el equipo de analistas solo puede revisar el 0.5 % diario, ese 0.5 % es el <code>contamination</code> que corresponde usar: es un presupuesto de alertas, no una estimación estadística.',
      watch: 'Mira los tiles de <b>Precisión / Recall / F1</b>, la línea de umbral en el <b>Histograma</b> y el punto operativo en <b>ROC / PR</b>.',
      target: 'card-hist',
      live: (s, f, ctx) => `Umbral s* = <b>${ctx.thr.toFixed(3)}</b> → marca <b>${ctx.flagged}</b> de ${ctx.n} puntos (${(100 * ctx.flagged / ctx.n).toFixed(1)}%). Las anomalías reales son ${ctx.pos} (${(100 * ctx.pos / ctx.n).toFixed(1)}%).`,
    },
    {
      id: 'maxFeatures', code: 'max_features', symbol: 'p', title: 'Columnas por árbol',
      short: 'Fracción (o número) de variables que cada árbol puede usar al elegir dónde cortar.',
      what: 'Cada árbol recibe un <b>subconjunto aleatorio de las columnas</b>. Con 1.0 todos los árboles pueden cortar en cualquier variable. Aquí los datos tienen solo 2 columnas, así que solo hay dos opciones: 1 de 2 (0.5) o 2 de 2 (1.0). En datos con muchas columnas el control es mucho más fino.',
      controls: 'Controla la <b>diversidad</b> entre árboles y su exposición a variables irrelevantes o a la ausencia de una variable clave. Con 1 columna por árbol, cada árbol solo hace cortes horizontales <i>o</i> verticales.',
      upLabel: 'Con más columnas por árbol', downLabel: 'Con menos columnas por árbol',
      up: 'Cada árbol ve toda la información: mejor detección cuando las anomalías solo se distinguen <b>combinando</b> columnas.',
      down: 'Más diversidad y velocidad, y útil si hay muchas columnas ruidosas. Pero un árbol sin la columna informativa aporta solo ruido: puede volverse ciego a anomalías (p. ej. un punto en el hueco de un anillo).',
      typical: '1.0 (todas). Reducirlo solo suele ayudar con muchas columnas (decenas o más) y bastante ruido; pruébalo, no lo supongas.',
      tune: { level: 'high', label: 'Prioridad alta (si hay muchas columnas)', text: 'En este laboratorio (2 columnas) es una decisión binaria con efecto enorme en el anillo.' },
      real: '<b>Ejemplo:</b> en manufactura, una pieza puede tener temperatura y presión "normales" por separado, pero una <i>combinación imposible</i>. Detectarla exige que cada árbol vea ambas variables, como en el anillo con hueco.',
      watch: 'Prueba el escenario del <b>anillo</b> y mira el mapa: con 0.5, el hueco central deja de detectarse.',
      target: 'card-map',
      live: (s, f) => `Cada árbol usa <b>${f.nFeatures} de ${f.d}</b> columnas${f.nFeatures < f.d ? ' (cortes en una sola dirección)' : ''}.`,
    },
    {
      id: 'bootstrap', code: 'bootstrap', symbol: '⟳', title: 'Muestreo con reemplazo',
      short: 'Si la submuestra de cada árbol se elige con reemplazo (un punto puede repetirse) o sin reemplazo (cada punto entra a lo sumo una vez).',
      what: 'Con <code>False</code> (por defecto) cada árbol recibe ψ puntos <b>distintos</b>. Con <code>True</code> se sortea con reemplazo (bootstrap): algunos puntos se repiten y otros no entran (≈ 63 % de puntos distintos cuando ψ = n).',
      controls: 'Controla el <b>tipo de aleatoriedad</b> del submuestreo. Los duplicados son puntos idénticos que <i>no se pueden separar entre sí</i>, así que actúan como un pequeño cúmulo denso dentro del árbol.',
      upLabel: 'Si lo activas (True)', downLabel: 'Si lo desactivas (False)',
      up: 'Más variación entre árboles, pero cada árbol ve menos puntos distintos; el efecto suele ser mínimo o ligeramente negativo.',
      down: 'Cada árbol aprovecha ψ puntos distintos: es lo recomendado por el paper original.',
      typical: '<code>False</code>. Es el hiperparámetro de menor impacto: úsalo para ver que <b>no todo hiperparámetro importa</b> (una idea central de la optimización, capítulo 4).',
      tune: { level: 'low', label: 'Prioridad baja', text: 'Casi nunca cambia el resultado más de lo que cambia la semilla.' },
      real: '<b>Ejemplo:</b> si tu tabla tiene registros duplicados por errores de captura, con <code>bootstrap=True</code> esos duplicados se multiplican y forman "cúmulos" artificiales; conviene deduplicar antes que ajustar este parámetro.',
      watch: 'Mira el <b>Barrido</b> (dos categorías casi iguales) y, en <i>Un árbol</i>, los puntos duplicados.',
      target: 'card-sweep',
      live: (s, f) => {
        let uniq = 0; const T = Math.min(f.trees.length, 50);
        for (let t = 0; t < T; t++) uniq += new Set(f.samples[t]).size / f.psi;
        return `Puntos distintos por árbol: <b>${(100 * uniq / T).toFixed(0)}%</b> de ψ.`;
      },
    },
    {
      id: 'seed', code: 'random_state', symbol: '#', title: 'Semilla aleatoria',
      short: 'Semilla del generador pseudoaleatorio. Isolation Forest es un algoritmo aleatorio: elige puntos, columnas y valores de corte al azar.',
      what: 'Fija la secuencia de números aleatorios. Misma semilla + mismos datos + mismos hiperparámetros ⇒ <b>exactamente el mismo bosque</b>. Aquí cada árbol usa una semilla derivada de (<i>random_state</i>, índice del árbol).',
      controls: 'Controla la <b>reproducibilidad</b>, no la calidad. Pero si cambiar la semilla altera mucho los resultados, es señal de que el modelo tiene demasiada varianza (pocos árboles o ψ pequeño).',
      upLabel: 'Buscar la "mejor" semilla', downLabel: 'Buena práctica',
      up: 'No existe "mejor" semilla: buscar la que da mejor métrica es hacer trampa (sobreajustas al azar).',
      down: 'Fijar la semilla para reproducir y <b>reportar media ± desviación</b> sobre varias semillas.',
      typical: 'Cualquier entero fijo (42, 0, …). Nunca la uses como hiperparámetro a optimizar.',
      tune: { level: 'no', label: 'Nunca optimizar', text: 'Se fija para reproducir; se varía solo para medir la varianza (estabilidad).' },
      real: '<b>Ejemplo:</b> en una auditoría te piden explicar por qué una transacción fue marcada hace tres meses. Sin semilla fija (y datos versionados) no puedes reproducir esa lista de alertas.',
      watch: 'Mira el <b>Barrido</b> de random_state: cada punto es una semilla distinta; su dispersión es la varianza del modelo. Y el tile de <b>Estabilidad</b>.',
      target: 'card-sweep',
      live: () => 'Con 1 solo árbol la semilla importa muchísimo; con 300 casi nada. Pruébalo.',
    },
  ];

  /* =====================================================================
   *  Métricas (tiles)
   * ===================================================================== */
  ML.IF_METRICS = {
    auc: {
      label: 'ROC-AUC', group: 'rank',
      tip: '<span class="tip-h">ROC-AUC</span>Probabilidad de que una anomalía real reciba un score <b>mayor</b> que un punto normal elegido al azar. 0.5 = azar, 1.0 = ranking perfecto.<span class="tip-sub">Necesita etiquetas. No depende del umbral (ni de contamination).</span>',
    },
    ap: {
      label: 'AP', group: 'rank',
      tip: '<span class="tip-h">Average Precision (área PR)</span>Resume la curva Precisión–Recall. Más exigente que el ROC-AUC cuando las anomalías son raras: una falsa alarma cuesta mucho.<span class="tip-sub">Necesita etiquetas. Un modelo al azar obtiene AP ≈ proporción de anomalías.</span>',
    },
    patk: {
      label: 'Precisión@k', group: 'rank',
      tip: '<span class="tip-h">Precisión@k</span>De las <b>k</b> alertas con mayor score (tu lista de revisión), qué fracción son anomalías reales. Refleja la vida real: el equipo solo revisa k casos.<span class="tip-sub">Necesita etiquetas. Elige k según tu capacidad de revisión.</span>',
    },
    precision: {
      label: 'Precisión', group: 'dec',
      tip: '<span class="tip-h">Precisión</span>De todo lo que el modelo <b>marcó</b> como anómalo, qué fracción lo era realmente. Baja = muchas falsas alarmas.<span class="tip-sub">Necesita etiquetas. Depende del umbral (contamination).</span>',
    },
    recall: {
      label: 'Recall', group: 'dec',
      tip: '<span class="tip-h">Recall (sensibilidad)</span>De todas las anomalías reales, qué fracción logró <b>detectar</b>. Bajo = se escapan anomalías.<span class="tip-sub">Necesita etiquetas. Depende del umbral (contamination).</span>',
    },
    f1: {
      label: 'F1', group: 'dec',
      tip: '<span class="tip-h">F1</span>Media armónica de precisión y recall: solo es alta si <b>ambas</b> lo son.<span class="tip-sub">Necesita etiquetas. Depende del umbral (contamination).</span>',
    },
    flagged: {
      label: 'Marcados', group: 'dec',
      tip: '<span class="tip-h">Puntos marcados</span>Cuántos puntos supera el umbral y se declaran anómalos, frente al total.',
    },
    jac: {
      label: 'Estabilidad Jaccard@k', group: 'unsup',
      tip: '<span class="tip-h">Estabilidad: Jaccard@k entre semillas</span>Entrenas el mismo modelo con 5 semillas y comparas las listas de las <b>k</b> alertas más altas: J(A,B) = |A∩B| / |A∪B|. 1 = siempre la misma lista; 0 = listas sin nada en común.<span class="tip-sub">NO necesita etiquetas. Mide varianza, no acierto: un modelo puede ser estable y estar equivocado.</span>',
    },
    spr: {
      label: 'Correlación de rangos', group: 'unsup',
      tip: '<span class="tip-h">Correlación de rangos (Spearman)</span>Cuánto coincide el <b>orden completo</b> de puntos por score entre semillas. 1 = mismo orden.<span class="tip-sub">NO necesita etiquetas. Es más sensible al orden de los puntos "normales" que Jaccard@k.</span>',
    },
    mv: {
      label: 'Mass-Volume (MV)', group: 'unsup', dir: -1,
      tip: '<span class="tip-h">Mass-Volume (Goix, 2016)</span>Volumen del <b>menor</b> conjunto de nivel del score que captura el 90–99.9 % de los datos. <b>Menor = mejor</b>: el modelo concentra la masa en poco espacio.<span class="tip-sub">NO necesita etiquetas. Se calcula en 2D por Monte Carlo. Puede fallar si las anomalías forman un cúmulo denso.</span>',
    },
    agree: {
      label: 'Acuerdo con kNN', group: 'unsup',
      tip: '<span class="tip-h">Acuerdo con otro detector</span>Jaccard@k entre tus alertas y las del detector de <b>distancia al 10.º vecino</b>. Si dos métodos muy distintos coinciden, es una señal de que hay estructura real.<span class="tip-sub">NO necesita etiquetas. Si el otro detector comparte el mismo punto ciego (p. ej. masking), el acuerdo engaña.</span>',
    },
  };

  /* =====================================================================
   *  Escenarios guiados (cargan el laboratorio)
   * ===================================================================== */
  ML.IF_LESSONS = [
    {
      id: 'trees', title: 'Más árboles: ¿hasta cuándo?', tag: 'n_estimators',
      setup: { dataset: 'blob', params: { nEstimators: 1, maxSamples: 256, contamination: 0.06, seed: 42, maxFeatures: 1, bootstrap: false }, active: 'nEstimators', target: 'card-sweep', sweepMetric: 'ap' },
      steps: ['Empieza con <b>1 árbol</b>: el mapa se ve irregular.', 'Sube a 3, 10, 50, 100 y por último 300.', 'Después de cada cambio mira el <b>Barrido</b> y la tabla (ms).'],
      look: 'El AP sube rápido al principio y luego se <b>aplana (plateau)</b>: las últimas duplicaciones de T casi no mueven la métrica, pero el <b>costo (ms)</b> sí crece. Ese es el mensaje: más árboles reduce el ruido, con rendimientos decrecientes; no es un hiperparámetro para "optimizar" sino para fijar en un valor suficiente.',
    },
    {
      id: 'contam', title: 'contamination mueve el umbral, no el ranking', tag: 'contamination',
      setup: { dataset: 'blob', params: { nEstimators: 100, maxSamples: 256, contamination: 'auto', seed: 42, maxFeatures: 1, bootstrap: false }, active: 'contamination', target: 'card-hist', sweepMetric: 'f1' },
      steps: ['Con <code>\'auto\'</code> el umbral es 0.5 y se marcan muchos puntos.', 'Cambia a un valor numérico y acércalo al % real de anomalías (≈ 6 %).'],
      look: '<b>ROC-AUC y AP no se mueven</b>: el ranking es el mismo. Lo que cambia es Precisión / Recall / F1, la línea de umbral en el histograma y el punto operativo sobre las curvas ROC y PR. Con un valor muy alto sube el recall y se hunde la precisión.',
    },
    {
      id: 'masking', title: 'Masking: submuestrear ayuda… hasta cierto punto', tag: 'max_samples',
      setup: { dataset: 'masking', params: { nEstimators: 100, maxSamples: 512, contamination: 0.06, seed: 42, maxFeatures: 1, bootstrap: false }, active: 'maxSamples', target: 'card-sweep', sweepMetric: 'ap' },
      steps: ['Con ψ = 512 (todos los puntos) el mini-cúmulo de anomalías queda en un rincón: <b>se enmascara</b>.', 'Baja ψ a 128 y luego a 64; después sigue bajando hasta 4.'],
      look: 'El AP <b>mejora al bajar ψ</b> porque el submuestreo dispersa el cúmulo anómalo y lo hace aislable… pero con ψ muy pequeño vuelve a empeorar: hay un <b>punto óptimo interior</b>. Repite en "Dos lunas": ahí ψ grande es mejor. No existe ψ universal (y por eso tiene sentido optimizarlo: capítulo 4).',
    },
    {
      id: 'ring', title: 'El anillo: cuando una columna no basta', tag: 'max_features',
      setup: { dataset: 'ring', params: { nEstimators: 100, maxSamples: 256, contamination: 0.06, seed: 42, maxFeatures: 1, bootstrap: false }, active: 'maxFeatures', target: 'card-map', sweepMetric: 'auc' },
      steps: ['Observa que el hueco central es una zona de score alto (fondo azul intenso): el modelo lo reconoce como anómalo.', 'Cambia <b>max_features</b> a 0.5 (1 de 2 columnas).', 'Fíjate en el tile de <b>Estabilidad Jaccard</b>: ¿empeora cuando el modelo empeora?'],
      look: 'Con una sola columna por árbol, un punto en el centro del anillo tiene un valor de x (o de y) idéntico al de muchos puntos normales del anillo: <b>no puede aislarse</b>. El AUC cae con fuerza y el hueco desaparece del mapa. Pero la <b>estabilidad no lo delata</b>: con los datos de este escenario pasa de 0.54 a 0.60 (¡sube!) mientras el ROC-AUC baja de 0.97 a 0.81. Un modelo estable no es un modelo correcto.',
    },
    {
      id: 'seed', title: 'random_state: ¿mala suerte o mal modelo?', tag: 'random_state',
      setup: { dataset: 'blob', params: { nEstimators: 3, maxSamples: 256, contamination: 0.06, seed: 1, maxFeatures: 1, bootstrap: false }, active: 'seed', target: 'card-sweep', sweepMetric: 'ap' },
      steps: ['Con solo 3 árboles cambia la semilla varias veces (botón 🎲) y observa las métricas.', 'Sube n_estimators a 200 y repite.'],
      look: 'En el <b>Barrido</b> de random_state cada punto es una semilla: con pocos árboles se dispersan mucho; con 200 casi se juntan. Si el resultado depende de la semilla, el problema no es la semilla: es la <b>varianza</b> del modelo (y la estabilidad Jaccard lo mide sin etiquetas).',
    },
    {
      id: 'fraud', title: 'Pocas anomalías: el caso del fraude', tag: 'caso real',
      setup: { dataset: 'blob', params: { nEstimators: 100, maxSamples: 256, contamination: 0.01, seed: 42, maxFeatures: 1, bootstrap: false }, active: 'contamination', target: 'card-hist', sweepMetric: 'f1', n: 1000, frac: 0.01 },
      steps: ['Aquí solo el 1 % son anomalías (en fraude real suele ser aún menos).', 'Compara el <b>ROC-AUC</b> con la <b>precisión media (AP)</b> y con <b>Precisión@k</b>.'],
      look: 'El ROC-AUC puede verse casi perfecto mientras la precisión con la que trabajaría tu equipo de analistas es mediocre: con muchísimos normales, una tasa pequeña de falsos positivos son <b>muchas</b> alertas falsas (' + cite('saito2015', 'Saito & Rehmsmeier, 2015') + '). Por eso en anomalías raras se mira AP y Precisión@k.',
    },
    {
      id: 'tree', title: 'Un árbol por dentro: así se aísla un punto', tag: 'concepto',
      setup: { dataset: 'blob', params: { nEstimators: 100, maxSamples: 64, contamination: 0.06, seed: 42, maxFeatures: 1, bootstrap: false }, active: 'maxSamples', target: 'card-map', view: 'tree' },
      steps: ['En la vista <b>Un árbol</b> mueve <i>Niveles de corte visibles</i> de 0 hacia arriba.', 'Haz clic en un punto normal del centro y luego en uno anómalo lejano.'],
      look: 'Un punto anómalo queda solo en su celda tras pocos cortes (h pequeño); uno del centro necesita muchos. El score del bosque es el promedio de esa profundidad sobre todos los árboles: <b>menos cortes ⇒ más anómalo</b>.',
    },
  ];

  /* =====================================================================
   *  Casos de la vida real
   * ===================================================================== */
  ML.IF_CASES = [
    {
      id: 'fraude', name: 'Fraude con tarjetas', kicker: 'Finanzas',
      ctx: 'Un banco procesa millones de transacciones y solo una fracción minúscula es fraude. En el conjunto público de la ULB (Worldline y Université Libre de Bruxelles): <b>492 fraudes en 284 807 transacciones = 0.172 %</b> (' + cite('dalpozzolo2015', 'Dal Pozzolo et al., 2015') + ').',
      goal: 'Producir una <b>lista corta de alertas</b> para que analistas la revisen cada día.',
      why: 'No necesita etiquetas para entrenar (los fraudes etiquetados llegan con semanas de retraso), escala a millones de filas y puntúa en milisegundos.',
      keys: [
        ['contamination', 'No uses <code>\'auto\'</code>: fíjalo por la <b>capacidad de revisión</b> (p. ej. el 0.5 % superior). Es un presupuesto de alertas.'],
        ['n_estimators', '100–200 bastan; más solo suma latencia en el scoring en tiempo real.'],
        ['max_samples', '256 es un buen inicio; valida 64–512 con las etiquetas retrasadas.'],
      ],
      measure: '<b>AP / PR-AUC</b> y <b>Precisión@k</b> (k = alertas que caben en el día). El ROC-AUC engaña con tasas tan bajas (' + cite('saito2015', 'Saito & Rehmsmeier, 2015') + ').',
      trap: 'El fraude evoluciona (deriva de datos): monitorea la <b>estabilidad</b> y reentrena. Las etiquetas para validar llegan tarde: valida con una ventana retrasada.',
      sim: { lesson: 'fraud', label: 'Simular pocas anomalías' },
    },
    {
      id: 'mantenimiento', name: 'Mantenimiento predictivo', kicker: 'Industria',
      ctx: 'Un compresor tiene decenas de sensores (vibración, temperatura, presión). Las fallas son raras y suelen verse como <b>combinaciones inusuales</b> de sensores, no como un valor fuera de rango.',
      goal: 'Detectar el estado "degradado" <b>antes</b> de la falla.',
      why: 'Sin etiquetas suficientes de fallas reales y con datos de sensores continuos, un detector no supervisado es el punto de partida natural.',
      keys: [
        ['max_samples', 'Cuando el estado degradado se repite, forma un <b>cúmulo denso</b> (masking): prueba ψ menores (32–128).'],
        ['max_features', '1.0 si la falla es una combinación de sensores; con muchas columnas ruidosas, valida valores menores.'],
        ['random_state', 'Fíjalo: el equipo de mantenimiento necesita alertas reproducibles.'],
      ],
      measure: 'Recall con <b>anticipación</b> suficiente, PR-AUC en fallas conocidas y la <b>estabilidad</b> entre reentrenos.',
      trap: 'El "modo de operación" cambia con la carga y la estación: una operación legítima nueva parecerá anómala. Segmenta por régimen.',
      sim: { lesson: 'masking', label: 'Simular masking' },
    },
    {
      id: 'ciber', name: 'Ciberseguridad y tráfico de red', kicker: 'Seguridad',
      ctx: 'Tráfico de red o logs de autenticación. El paper original evaluó Isolation Forest, entre otros, con tráfico de red (subconjunto <i>Http</i> de KDD CUP 99) y ForestCover (' + cite('liu2008', 'Liu et al., 2008') + ').',
      goal: 'Priorizar eventos sospechosos para el equipo de seguridad.',
      why: 'Rápido, con bajo consumo de memoria y sin depender de firmas de ataques conocidos.',
      keys: [
        ['max_samples', 'Un ataque masivo genera miles de eventos casi idénticos: un <b>cúmulo denso de anomalías</b>. Con dos cúmulos anómalos densos, el AUC pasó de 0.67 (todos los datos) a 0.91 con ψ = 128 (' + cite('liu2012', 'Liu et al., 2012') + ').'],
        ['n_estimators', 'Estabiliza y fija; no lo optimices.'],
      ],
      measure: 'Precisión@k (capacidad del SOC), estabilidad de la lista y cobertura de tipos de ataque conocidos.',
      trap: 'El atacante adapta su comportamiento y el "normal" deriva. Un detector sin contexto genera fatiga de alertas.',
      sim: { lesson: 'masking', label: 'Simular masking' },
    },
    {
      id: 'calidad', name: 'Calidad en manufactura', kicker: 'Industria',
      ctx: 'Cada pieza se mide con dos variables correlacionadas (p. ej. temperatura y presión de una prensa). Cada valor por separado es normal, pero la <b>combinación</b> es imposible: forma un "hueco" rodeado de piezas normales.',
      goal: 'Encontrar piezas con combinaciones anómalas de medidas.',
      why: 'Isolation Forest puede detectar anomalías rodeadas de puntos normales, como el anillo (' + cite('liu2012', 'Liu et al., 2012') + ').',
      keys: [
        ['max_features', '<b>1.0</b>: cada árbol necesita ver ambas variables para reconocer la combinación anómala.'],
        ['max_samples', 'Valida un ψ mayor si la forma de la región normal es compleja.'],
      ],
      measure: 'Mapa de scores (visual), AP y revisión de las piezas marcadas por un ingeniero de proceso.',
      trap: 'Los cortes paralelos a los ejes generan artefactos en los mapas de score; existe una variante con cortes oblicuos (' + cite('hariri2021', 'Hariri et al., 2021') + ').',
      sim: { lesson: 'ring', label: 'Simular el anillo' },
    },
    {
      id: 'salud', name: 'Monitoreo clínico', kicker: 'Salud',
      ctx: 'Perfiles de signos vitales o análisis de laboratorio en los que se buscan pacientes con patrones atípicos para <b>revisión clínica</b>. El diagnóstico médico figura entre las aplicaciones típicas de la detección de anomalías (' + cite('domingues2018', 'Domingues et al., 2018') + ').',
      goal: 'Levantar casos atípicos para que un profesional los revise.',
      why: 'Los casos graves son pocos y heterogéneos; no hay etiquetas suficientes de "todos los tipos de anormalidad".',
      keys: [
        ['contamination', 'Aquí un falso negativo cuesta mucho más que un falso positivo: el umbral se fija para <b>recall alto</b> con una precisión mínima aceptable. Es una decisión clínica, no estadística.'],
        ['random_state', 'La trazabilidad de qué caso se marcó y por qué es un requisito.'],
      ],
      measure: 'Recall a una precisión mínima; estabilidad de la lista; revisión clínica de una muestra.',
      trap: 'El score es una <b>priorización</b>, no un diagnóstico. Sesgos en los datos históricos se heredan.',
      sim: { lesson: 'contam', label: 'Simular el umbral' },
    },
    {
      id: 'limpieza', name: 'Limpieza de datos', kicker: 'Ciencia de datos',
      ctx: 'Antes de entrenar un modelo supervisado quieres encontrar registros erróneos: sensores caídos, capturas mal digitadas, unidades mezcladas. La limpieza de datos es otra aplicación clásica (' + cite('domingues2018', 'Domingues et al., 2018') + ').',
      goal: 'Producir una lista de registros a <b>auditar</b>, no a borrar automáticamente.',
      why: 'No hace falta etiquetar errores; sirve como filtro de revisión.',
      keys: [
        ['contamination', 'Estímalo con una auditoría de una muestra pequeña: el % de registros erróneos que encuentres.'],
        ['random_state / n_estimators', 'La lista de registros a revisar <b>no puede cambiar</b> cada vez que la corres: mide la estabilidad Jaccard.'],
      ],
      measure: 'Estabilidad Jaccard@k y la tasa de errores reales hallados en la muestra auditada.',
      trap: 'Borrar automáticamente las "anomalías" puede eliminar casos raros pero legítimos (justo los que más importan).',
      sim: { lesson: 'trees', label: 'Simular estabilidad' },
    },
  ];

  /* =====================================================================
   *  Preguntas de comprobación (por capítulo)
   * ===================================================================== */
  ML.IF_CHECKS = {
    ch6: [
      { q: 'Te piden "el mejor modelo" pero no tienes etiquetas. ¿Cuáles son tus tres primeros pasos?', a: '1) Definir la <b>capacidad de revisión (k)</b> y el costo de los errores; 2) <b>fijar</b> T y la semilla y medir la <b>estabilidad</b>; 3) conseguir una pequeña muestra etiquetada, <b>inyectar anomalías sintéticas</b> o pedir revisión experta del top-k para poder validar.' },
      { q: '¿Qué monitorearías después de desplegar el detector?', a: 'La <b>estabilidad</b> entre reentrenos (Jaccard), la <b>distribución de scores</b> (deriva), la <b>tasa de alertas confirmadas</b> por los analistas y los cambios en los datos de entrada.' },
    ],
    ch1: [
      { q: 'Un punto anómalo se aísla con menos cortes que uno normal. ¿Por qué, sin usar la palabra "distancia"?', a: 'Porque las anomalías son <b>pocas y diferentes</b>: están en regiones casi vacías, así que un corte aleatorio suele dejar al punto solo en una de las dos mitades. Los puntos normales están rodeados de muchos parecidos y necesitan muchos cortes para quedar solos.' },
      { q: 'Un score s(x) = 0.5 para casi todos los puntos, ¿qué te dice?', a: 'Que el modelo <b>no encuentra estructura</b>: ningún punto se aísla claramente antes que los demás (E[h(x)] ≈ c(ψ)).' },
    ],
    ch2: [
      { q: 'Subes <code>n_estimators</code> de 100 a 300: el AP pasa de 0.921 a 0.925 y el tiempo se triplica. ¿Qué concluyes?', a: 'Que estás en el <b>plateau</b>: la mejora (+0.004) es menor que la variación entre semillas y cuesta 3× más. Lo razonable es quedarse en ~100 y no tratarlo como hiperparámetro a optimizar.' },
      { q: 'Cambias <code>contamination</code> de 0.06 a 0.15 y el ROC-AUC no se mueve. ¿Es un error?', a: 'No: <code>contamination</code> solo fija el <b>umbral</b>; el bosque y el ranking son idénticos. Cambian Precisión/Recall/F1 y el número de puntos marcados.' },
    ],
    ch3: [
      { q: 'Tu modelo tiene estabilidad Jaccard@25 = 0.98 entre semillas. ¿Está bien?', a: 'Solo sabes que es <b>estable</b>, no que sea <b>correcto</b>. En el anillo con <code>max_features=0.5</code> el modelo es tan estable como el correcto (incluso algo más) y está ciego al hueco. La estabilidad es condición necesaria, no suficiente.' },
      { q: 'No tienes etiquetas. ¿Qué combinas para decidir entre dos configuraciones?', a: 'Varios indicadores a la vez: <b>estabilidad</b>, Mass-Volume, acuerdo con otro detector y, sobre todo, la <b>revisión manual</b> del top-k por un experto. Ninguno solo es fiable (mira el fallo en "masking").' },
    ],
    ch4: [
      { q: 'Tienes presupuesto para 20 evaluaciones y 3 hiperparámetros, uno de los cuales casi no importa. ¿Grid o random?', a: '<b>Random</b> (o bayesiano): con una malla de 20 puntos solo pruebas unos pocos valores de cada hiperparámetro importante; con puntos aleatorios pruebas 20 valores distintos de cada uno (' + cite('bergstra2012', 'Bergstra & Bengio, 2012') + ').' },
      { q: 'Optimizaste con Mass-Volume y la configuración elegida rinde peor de lo esperado. ¿Qué pudo pasar?', a: 'Que los indicadores sin etiquetas tienen <b>puntos ciegos</b>: con anomalías agrupadas en un cúmulo denso, MV las considera "normales". Valida siempre con un pequeño conjunto etiquetado o anomalías inyectadas.' },
    ],
    ch5: [
      { q: 'En fraude con 0.17 % de anomalías, ¿por qué no basta el ROC-AUC?', a: 'Porque con muchísimos normales, una tasa pequeña de falsos positivos son <b>muchas</b> alertas falsas que el ROC no muestra. Precisión, AP y Precisión@k sí las reflejan (' + cite('saito2015', 'Saito & Rehmsmeier, 2015') + ').' },
    ],
  };

  ML.IF_TAKEAWAYS = {
    ch1: ['Las anomalías se aíslan con <b>pocos cortes</b> aleatorios; los puntos normales necesitan muchos.', 'El score s(x) = 2<sup>−E[h]/c(ψ)</sup>: ≈ 1 anómalo, ≈ 0.5 sin estructura, ≈ 0 normal.', 'El azar del algoritmo tiene dos fuentes: <b>qué puntos</b> ve cada árbol y <b>dónde corta</b>.'],
    ch2: ['<code>n_estimators</code>, <code>max_samples</code>, <code>max_features</code> y <code>bootstrap</code> cambian el <b>ranking</b>; <code>contamination</code> solo cambia el <b>umbral</b>.', 'Más árboles reduce el ruido con <b>rendimientos decrecientes</b>; el mejor ψ y max_features <b>dependen de los datos</b> (masking, anillo).', 'La semilla no es un hiperparámetro: si cambiarla altera el resultado, el problema es la <b>varianza</b> del modelo.'],
    ch3: ['Con etiquetas: <b>AP y Precisión@k</b> (el ROC-AUC engaña con anomalías muy raras).', 'Sin etiquetas: estabilidad (Jaccard), Mass-Volume y consenso ayudan, pero cada uno tiene <b>puntos ciegos</b>.', '<b>Estable ≠ correcto</b>: combina indicadores y valida con revisión de un experto.'],
    ch4: ['Vale la pena tunear lo que depende de los datos (<code>max_samples</code>, <code>max_features</code>); <b>no</b> T, la semilla ni contamination.', 'Random, bayesiana y halving superan a grid cuando pocos hiperparámetros importan: compáralos por su <b>costo</b>.', 'Sin etiquetas, el objetivo del tuner es un <b>proxy</b>: puede llevarte con confianza a una mala configuración.'],
    ch5: ['Cada hiperparámetro se traduce en una <b>decisión de negocio</b> (capacidad de revisión, costo de errores, reproducibilidad).', 'El indicador correcto depende de <b>cuántas alertas</b> puede revisar el equipo.', 'Anticipa la trampa del dominio: deriva de datos, masking, etiquetas tardías.'],
  };

  /* =====================================================================
   *  Glosario (tooltips automáticos en <span data-term="…">)
   * ===================================================================== */
  ML.GLOSSARY = {
    anomalia: ['Anomalía', 'Observación que se aleja de la mayoría de los datos. Suele ser <b>rara y diferente</b>. No siempre es un error: puede ser el hallazgo que buscas.'],
    score: ['Score de anomalía s(x)', 'Número entre 0 y 1: s(x) = 2<sup>−E[h(x)]/c(ψ)</sup>. Cerca de 1 = muy anómalo; ≈ 0.5 = sin estructura; cerca de 0 = muy normal.'],
    umbral: ['Umbral de decisión', 'Valor del score a partir del cual un punto se declara anómalo. Lo fija <code>contamination</code>.'],
    hiperparametro: ['Hiperparámetro', 'Ajuste del <b>algoritmo</b> que tú eliges antes de entrenar (p. ej. nº de árboles). Se distingue de los <i>parámetros</i>, que el algoritmo aprende de los datos.'],
    ensemble: ['Ensemble / bosque', 'Conjunto de muchos modelos simples cuyos resultados se combinan (aquí, se promedian las profundidades).'],
    ruido: ['Varianza / ruido', 'Cuánto cambia el resultado solo por el azar del algoritmo (otra semilla). Más árboles la reducen.'],
    masking: ['Masking', 'Anomalías muy agrupadas que se "esconden" entre sí y parecen un cúmulo normal; cuesta aislarlas.'],
    swamping: ['Swamping', 'Puntos normales cercanos a las anomalías que se confunden con ellas.'],
    aucroc: ['ROC-AUC', 'Probabilidad de que una anomalía real puntúe más alto que un normal al azar. Necesita etiquetas.'],
    ap: ['Precisión media (AP)', 'Área bajo la curva Precisión–Recall. Mejor que ROC-AUC cuando las anomalías son raras. Necesita etiquetas.'],
    precision: ['Precisión', 'De lo que marcaste, qué fracción era anomalía real.'],
    recall: ['Recall', 'De las anomalías reales, qué fracción detectaste.'],
    jaccard: ['Índice de Jaccard', 'J(A,B) = |A∩B| / |A∪B|. Mide cuánto se parecen dos <b>listas</b> (1 = idénticas, 0 = nada en común).'],
    spearman: ['Correlación de Spearman', 'Correlación entre los <b>rangos</b> (posiciones) de dos rankings. 1 = mismo orden.'],
    mv: ['Mass-Volume', 'Criterio sin etiquetas (Goix, 2016): volumen mínimo de un conjunto de nivel del score que contiene una fracción α de los datos.'],
    plateau: ['Plateau', 'Zona en la que seguir aumentando un valor (p. ej. nº de árboles) ya no mejora la métrica de forma medible.'],
    tunabilidad: ['Tunabilidad', 'Cuánto rendimiento se gana <b>optimizando</b> un hiperparámetro frente a su valor por defecto (Probst et al., 2019).'],
    grid: ['Grid search', 'Búsqueda exhaustiva en una malla de valores predefinidos. El costo crece exponencialmente con el nº de hiperparámetros.'],
    random: ['Random search', 'Búsqueda que muestrea configuraciones al azar del espacio. Sorprendentemente eficiente cuando pocos hiperparámetros importan.'],
    bayes: ['Optimización bayesiana', 'Construye un modelo (p. ej. proceso gaussiano) de cómo el hiperparámetro afecta la métrica y elige la siguiente configuración que más promete.'],
    halving: ['Successive Halving', 'Evalúa muchas configuraciones con poco recurso (p. ej. pocos árboles) y solo promueve la mejor fracción a más recurso.'],
    fidelidad: ['Fidelidad / recurso', 'Cuánto "esfuerzo" se invierte al evaluar una configuración (aquí: nº de árboles). Baja fidelidad = barata pero ruidosa.'],
    incumbente: ['Incumbente', 'La mejor configuración encontrada hasta el momento por el optimizador.'],
    leakage: ['Fuga de información (leakage)', 'Cuando datos de validación o test influyen en el entrenamiento o la selección; produce métricas optimistas.'],
    pr: ['Curva Precisión–Recall', 'Precisión frente a recall para todos los umbrales posibles. El punto naranja marca el umbral actual.'],
    knn: ['kNN (distancia al k-ésimo vecino)', 'Detector simple: cuanto más lejos está el 10.º vecino de un punto, más anómalo es.'],
    semilla: ['Semilla (random_state)', 'Número que fija el generador aleatorio: misma semilla ⇒ mismo bosque.'],
    presupuesto: ['Presupuesto', 'Cuánto cómputo puedes gastar buscando hiperparámetros (aquí: nº de árboles entrenados).'],
    patk: ['Precisión@k', 'De las k alertas con mayor score, qué fracción son anomalías reales.'],
  };
})();
