/* Contenido pedagógico del laboratorio de Isolation Forest (definiciones, escenarios y métricas). */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});

  /** Hiperparámetros. Cada uno incluye definición, rol en el modelo y efecto de subirlo/bajarlo. */
  ML.IF_PARAMS = [
    {
      id: 'nEstimators', code: 'n_estimators', symbol: 'T', title: 'Número de árboles',
      short: 'Cuántos árboles de aislamiento independientes forman el bosque. Cada árbol "opina" una profundidad de aislamiento y el bosque promedia todas las opiniones.',
      what: 'El bosque es un <b>conjunto (ensemble)</b> de <i>T</i> árboles construidos con azar. Cada árbol es un estimador muy ruidoso de qué tan fácil es aislar un punto; el score final usa el <b>promedio</b> de las profundidades, <i>E[h(x)]</i>.',
      controls: 'Controla la <b>varianza</b> del score, no su sesgo: el error del promedio decrece aproximadamente como <i>1/√T</i>. No cambia lo que el modelo "piensa" en promedio; cambia cuánta suerte interviene.',
      up: 'Scores más estables y reproducibles; el ranking converge. El costo (tiempo y memoria) crece linealmente y hay <b>rendimientos decrecientes</b>. No existe sobreajuste por tener más árboles.',
      down: 'Scores ruidosos: dos ejecuciones con distinta semilla dan rankings distintos. Con muy pocos árboles un punto normal puede "tener suerte" y aislarse pronto (falsa alarma).',
      typical: '100 (valor por defecto de scikit-learn). Entre 100 y 300 casi siempre basta; súbelo si necesitas resultados muy estables.',
      watch: 'Mira <b>Convergencia</b> (las líneas se aplanan) y el <b>Barrido</b> (la banda entre semillas se estrecha).',
      target: 'card-conv',
      live: (s, f) => `Con T = ${f.nEstimators} el ruido del promedio es ≈ 1/√T = <b>${(1 / Math.sqrt(f.nEstimators)).toFixed(2)}</b> veces el de un solo árbol.`,
    },
    {
      id: 'maxSamples', code: 'max_samples', symbol: 'ψ', title: 'Tamaño de submuestra',
      short: 'Cuántos puntos (elegidos al azar) usa cada árbol para construirse. No usa todos los datos: submuestrear es parte esencial del algoritmo.',
      what: 'Cada árbol se entrena con una <b>submuestra aleatoria</b> de ψ puntos. La profundidad máxima se deriva de ψ: <code>⌈log₂ ψ⌉</code> (no hace falta profundizar más: las anomalías se aíslan antes). ψ también normaliza el score mediante <i>c(ψ)</i>.',
      controls: 'Controla la <b>resolución</b> de cada árbol y combate dos problemas: el <b>swamping</b> (normales cercanos a anomalías se confunden con ellas) y el <b>masking</b> (anomalías agrupadas se "esconden" entre sí). Muestras pequeñas dispersan los grupos de anomalías y las hacen aislables.',
      up: 'Árboles más profundos y con más detalle de la forma de los datos, pero más masking/swamping y más costo. Ganas resolución en estructuras complejas (p. ej. lunas).',
      down: 'Árboles más rápidos, diversos y resistentes al masking; pero con ψ muy pequeño (&lt; 32) los árboles son demasiado toscos y los scores se vuelven imprecisos.',
      typical: '256 (paper original y scikit-learn). <b>No hay valor universal</b>: con anomalías agrupadas conviene ψ menor (32–128); con estructuras complejas y sin masking, mayor.',
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
      typical: 'Úsalo con conocimiento del negocio (p. ej. "≈ 1 % de las transacciones son fraude"). Si no lo conoces, prefiere trabajar con el <b>score / ranking</b> (top-k) en lugar de fijar un porcentaje.',
      watch: 'Mira los tiles de <b>Precisión / Recall / F1</b>, la línea de umbral en el <b>Histograma</b> y el punto operativo en <b>ROC / PR</b>.',
      target: 'card-hist',
      live: (s, f, ctx) => `Umbral s* = <b>${ctx.thr.toFixed(3)}</b> → marca <b>${ctx.flagged}</b> de ${ctx.n} puntos (${(100 * ctx.flagged / ctx.n).toFixed(1)}%). Las anomalías reales son ${ctx.pos} (${(100 * ctx.pos / ctx.n).toFixed(1)}%).`,
    },
    {
      id: 'maxFeatures', code: 'max_features', symbol: 'p', title: 'Columnas por árbol', upLabel: 'Con más columnas por árbol', downLabel: 'Con menos columnas por árbol',
      short: 'Fracción (o número) de variables que cada árbol puede usar al elegir dónde cortar.',
      what: 'Cada árbol recibe un <b>subconjunto aleatorio de las columnas</b>. Con 1.0 todos los árboles pueden cortar en cualquier variable. Aquí los datos tienen solo 2 columnas, así que solo hay dos opciones: 1 de 2 (0.5) o 2 de 2 (1.0). En datos con muchas columnas el control es mucho más fino.',
      controls: 'Controla la <b>diversidad</b> entre árboles y su exposición a variables irrelevantes o a la ausencia de una variable clave. Con 1 columna por árbol, cada árbol solo hace cortes horizontales <i>o</i> verticales.',
      up: 'Cada árbol ve toda la información: mejor detección cuando las anomalías solo se distinguen combinando columnas.',
      down: 'Más diversidad y velocidad, y útil si hay muchas columnas ruidosas. Pero un árbol sin la columna informativa aporta solo ruido: puede volverse ciego a anomalías (p. ej. un punto en el hueco de un anillo).',
      typical: '1.0 (todas). Reducirlo solo suele ayudar con muchas columnas (decenas o más) y bastante ruido.',
      watch: 'Prueba el escenario del <b>anillo</b> y mira el mapa: con 0.5, el hueco central deja de detectarse.',
      target: 'card-map',
      live: (s, f) => `Cada árbol usa <b>${f.nFeatures} de ${f.d}</b> columnas${f.nFeatures < f.d ? ' (cortes en una sola dirección)' : ''}.`,
    },
    {
      id: 'bootstrap', code: 'bootstrap', symbol: '⟳', title: 'Muestreo con reemplazo', upLabel: 'Si lo activas (True)', downLabel: 'Si lo desactivas (False)',
      short: 'Si la submuestra de cada árbol se elige con reemplazo (un punto puede repetirse) o sin reemplazo (cada punto entra a lo sumo una vez).',
      what: 'Con <code>False</code> (por defecto) cada árbol recibe ψ puntos <b>distintos</b>. Con <code>True</code> se sortea con reemplazo (bootstrap): algunos puntos se repiten y otros no entran (≈ 63 % de puntos distintos cuando ψ = n).',
      controls: 'Controla el <b>tipo de aleatoriedad</b> del submuestreo. Los duplicados son puntos idénticos que <i>no se pueden separar entre sí</i>, así que actúan como un pequeño cúmulo denso dentro del árbol.',
      up: '(Activar) Más variación entre árboles, pero cada árbol ve menos puntos distintos; el efecto suele ser mínimo o ligeramente negativo.',
      down: '(Desactivar) Cada árbol aprovecha ψ puntos distintos: es lo recomendado por el paper original.',
      typical: '<code>False</code>. Es el hiperparámetro de menor impacto: úsalo para ver que <b>no todo hiperparámetro importa</b>.',
      watch: 'Mira el <b>Barrido</b> (dos categorías casi iguales) y, en <i>Un árbol</i>, los puntos duplicados.',
      target: 'card-sweep',
      live: (s, f) => {
        let uniq = 0; const T = Math.min(f.trees.length, 50);
        for (let t = 0; t < T; t++) uniq += new Set(f.samples[t]).size / f.psi;
        return `Puntos distintos por árbol: <b>${(100 * uniq / T).toFixed(0)}%</b> de ψ.`;
      },
    },
    {
      id: 'seed', code: 'random_state', symbol: '#', title: 'Semilla aleatoria', upLabel: 'Buscar la "mejor" semilla', downLabel: 'Buena práctica',
      short: 'Semilla del generador pseudoaleatorio. Isolation Forest es un algoritmo aleatorio: elige puntos, columnas y valores de corte al azar.',
      what: 'Fija la secuencia de números aleatorios. Misma semilla + mismos datos + mismos hiperparámetros ⇒ <b>exactamente el mismo bosque</b>. Aquí cada árbol usa una semilla derivada de (<i>random_state</i>, índice del árbol).',
      controls: 'Controla la <b>reproducibilidad</b>, no la calidad. Pero si cambiar la semilla altera mucho los resultados, es señal de que el modelo tiene demasiada varianza (pocos árboles o ψ pequeño).',
      up: 'No existe "mejor" semilla: buscar la que da mejor métrica es hacer trampa (sobreajustas al azar).',
      down: 'Ídem. Buena práctica: fijar la semilla para reproducir y <b>reportar media ± desviación</b> sobre varias semillas.',
      typical: 'Cualquier entero fijo (42, 0, …). Nunca la uses como hiperparámetro a optimizar.',
      watch: 'Mira el <b>Barrido</b> de random_state: cada punto es una semilla distinta; su dispersión es la varianza del modelo.',
      target: 'card-sweep',
      live: () => 'Con 1 solo árbol la semilla importa muchísimo; con 300 casi nada. Pruébalo.',
    },
  ];

  /** Definiciones de las métricas (para tiles y tooltips). */
  ML.IF_METRICS = {
    auc: {
      label: 'ROC-AUC', group: 'rank',
      tip: '<span class="tip-h">ROC-AUC</span>Probabilidad de que una anomalía real reciba un score <b>mayor</b> que un punto normal elegido al azar. 0.5 = azar, 1.0 = ranking perfecto.<span class="tip-sub">No depende del umbral (ni de contamination).</span>',
    },
    ap: {
      label: 'Precisión media (AP)', group: 'rank',
      tip: '<span class="tip-h">Average Precision (área PR)</span>Resume la curva Precisión–Recall. Es más exigente que el ROC-AUC cuando las anomalías son raras: una falsa alarma cuesta mucho.<span class="tip-sub">Un modelo al azar obtiene AP ≈ proporción de anomalías.</span>',
    },
    precision: {
      label: 'Precisión', group: 'dec',
      tip: '<span class="tip-h">Precisión</span>De todo lo que el modelo <b>marcó</b> como anómalo, qué fracción lo era realmente. Baja = muchas falsas alarmas.<span class="tip-sub">Depende del umbral (contamination).</span>',
    },
    recall: {
      label: 'Recall', group: 'dec',
      tip: '<span class="tip-h">Recall (sensibilidad)</span>De todas las anomalías reales, qué fracción logró <b>detectar</b>. Bajo = se escapan anomalías.<span class="tip-sub">Depende del umbral (contamination).</span>',
    },
    f1: {
      label: 'F1', group: 'dec',
      tip: '<span class="tip-h">F1</span>Media armónica de precisión y recall: solo es alta si <b>ambas</b> lo son.<span class="tip-sub">Depende del umbral (contamination).</span>',
    },
    flagged: {
      label: 'Marcados', group: 'dec',
      tip: '<span class="tip-h">Puntos marcados</span>Cuántos puntos supera el umbral y se declaran anómalos, frente al total.',
    },
  };

  /** Escenarios guiados: cargan datos + hiperparámetros y dicen qué observar. */
  ML.IF_LESSONS = [
    {
      id: 'trees', title: 'Más árboles, más estabilidad', tag: 'n_estimators',
      setup: { dataset: 'blob', params: { nEstimators: 1, maxSamples: 256, contamination: 0.06, seed: 42, maxFeatures: 1, bootstrap: false }, active: 'nEstimators', target: 'card-sweep', sweepMetric: 'ap' },
      steps: ['Empieza con <b>1 árbol</b>: el mapa se ve irregular.', 'Sube a 3, 10, 50 y 300.'],
      look: 'En el <b>Barrido</b> la precisión media (AP) sube y la banda entre semillas se estrecha; en <b>Convergencia</b> las líneas se aplanan pasadas las primeras decenas de árboles. Más allá de ~100 el beneficio es marginal.',
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
      look: 'El AP <b>mejora al bajar ψ</b> porque el submuestreo dispersa el cúmulo anómalo y lo hace aislable… pero con ψ muy pequeño vuelve a empeorar: hay un <b>punto óptimo</b>. Repite en el dataset "Dos lunas": ahí ψ grande es mejor. No existe ψ universal.',
    },
    {
      id: 'ring', title: 'El anillo: cuando una columna no basta', tag: 'max_features',
      setup: { dataset: 'ring', params: { nEstimators: 100, maxSamples: 256, contamination: 0.06, seed: 42, maxFeatures: 1, bootstrap: false }, active: 'maxFeatures', target: 'card-map', sweepMetric: 'auc' },
      steps: ['Observa que el hueco central es una zona de score alto (fondo azul intenso): el modelo lo reconoce como anómalo.', 'Cambia <b>max_features</b> a 0.5 (1 de 2 columnas).'],
      look: 'Con una sola columna por árbol, un punto en el centro del anillo tiene un valor de x (o de y) idéntico al de muchos puntos normales del anillo: <b>no puede aislarse</b>. El AUC cae con fuerza y el hueco desaparece del mapa. Cada árbol necesita ver <i>ambas</i> dimensiones para reconocer la combinación anómala.',
    },
    {
      id: 'seed', title: 'random_state: ¿mala suerte o mal modelo?', tag: 'random_state',
      setup: { dataset: 'blob', params: { nEstimators: 3, maxSamples: 256, contamination: 0.06, seed: 1, maxFeatures: 1, bootstrap: false }, active: 'seed', target: 'card-sweep', sweepMetric: 'ap' },
      steps: ['Con solo 3 árboles cambia la semilla varias veces (botón 🎲) y observa las métricas.', 'Sube n_estimators a 200 y repite.'],
      look: 'En el <b>Barrido</b> de random_state cada punto es una semilla: con pocos árboles se dispersan mucho; con 200 casi se juntan. Si el resultado depende de la semilla, el problema no es la semilla: es la <b>varianza</b> del modelo.',
    },
    {
      id: 'tree', title: 'Un árbol por dentro: así se aísla un punto', tag: 'concepto',
      setup: { dataset: 'blob', params: { nEstimators: 100, maxSamples: 64, contamination: 0.06, seed: 42, maxFeatures: 1, bootstrap: false }, active: 'maxSamples', target: 'card-map', view: 'tree' },
      steps: ['En la vista <b>Un árbol</b> mueve <i>Niveles de corte visibles</i> de 0 hacia arriba.', 'Haz clic en un punto normal del centro y luego en uno anómalo lejano.'],
      look: 'Un punto anómalo queda solo en su celda tras pocos cortes (h pequeño); uno del centro necesita muchos. El score del bosque es el promedio de esa profundidad sobre todos los árboles: <b>menos cortes ⇒ más anómalo</b>.',
    },
  ];
})();
