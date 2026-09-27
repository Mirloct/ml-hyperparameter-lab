/* Contenido pedagógico del laboratorio de VAE. Las citas (Autor, año) enlazan con la bibliografía. */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});
  const cite = (id, txt) => `<a class="cite" href="#ref-${id}">${txt}</a>`;
  ML.cite = cite;

  /* Conceptos que el verificador de secuencia de lectura (tools/reading-check.js) comprueba:
   * cada uno debe estar PRESENTADO (glosario, encabezado, tabla…) en su primera aparición. */
  ML.READING_CONCEPTS = [
    ['autoencoder', 'autoencoder', 'ch1'], ['espacio latente', 'latente', 'ch1'], ['encoder', 'encoder', 'ch1'],
    ['decoder', 'decoder', 'ch1'], ['KL', 'KL', 'ch1'], ['prior', 'prior', 'ch1'], ['ELBO', 'ELBO', 'ch1'],
    ['reparametrización', 'reparametrizaci', 'ch1'], ['nats', 'nats', 'ch1'], ['reconstrucción', 'reconstrucci', 'ch1'],
    ['hiperparámetro', 'hiperparámetro', 'top'], ['colapso posterior', 'colapso', 'ch2'], ['recocido', 'recocido', 'ch2'],
    ['desenredo', 'desenredo', 'ch2'], ['época', 'época', 'ch2'], ['lote', 'lote', 'ch2'], ['Adam', 'Adam', 'ch2'],
    ['plateau', 'plateau', 'ch2'], ['sobreajuste', 'sobreajust', 'ch2'], ['generativo', 'generativo', 'ch2'],
    ['PCA', 'PCA', 'ch2'], ['ROC-AUC', 'ROC-AUC', 'ch3'], ['AP', 'AP', 'ch3'],
  ];

  /* =====================================================================
   *  Hiperparámetros
   * ===================================================================== */
  ML.VAE_PARAMS = [
    {
      id: 'latent', code: 'latent_dim', symbol: 'L', title: 'Dimensión latente',
      short: 'Cuántos números usa el modelo para describir cada observación. Es el ancho del "cuello de botella".',
      what: 'El VAE comprime cada entrada en <b>L</b> números (la media μ de una distribución) y la reconstruye desde ahí. Con L = 2 cada imagen se resume en dos coordenadas; con L = 8, en ocho.',
      controls: 'Controla la <b>capacidad de información</b> del código. Si L es menor que el número de factores reales que generan los datos, el modelo <b>tiene que descartar</b> alguno.',
      upLabel: 'Si la subes', downLabel: 'Si la bajas',
      up: 'Más capacidad para reconstruir bien. Pero el VAE <b>no usa todas</b> las dimensiones que le des: las que no aportan se apagan solas (KL ≈ 0). Medido aquí: con L = 8 en datos de 2 factores, solo 3 quedan activas.',
      down: 'Compresión más agresiva y un espacio más fácil de visualizar, pero si te quedas corto pierdes información de forma irreversible. Medido aquí: en «Punto: posición y tamaño» (3 factores) con L = 2, el <b>tamaño deja de codificarse</b> (correlación 0.08).',
      typical: 'Empieza en 2 para poder dibujarlo; en datos reales, entre 8 y 64. Súbela hasta que el error de reconstrucción deje de mejorar y mira cuántas dimensiones quedan <b>activas</b>: ese es el número que los datos piden.',
      tune: { level: 'high', label: 'Prioridad alta' },
      real: '<b>Ejemplo:</b> en control de calidad con 200 sensores, si las piezas varían realmente por 5 causas físicas, un latente de 5–10 basta. Poner 100 no mejora y hace el espacio inútil para inspeccionar.',
      watch: 'Mira <b>KL por dimensión</b> (cuántas barras sobreviven) y el <b>Barrido</b> de latent_dim.',
      target: 'card-kl',
      live: (s, m, ev) => `L = <b>${m.L}</b>, de las cuales <b>${ev.active}</b> están activas (KL > 0.01). Los datos tienen <b>${s.nF}</b> factores reales.`,
    },
    {
      id: 'beta', code: 'beta (β)', symbol: 'β', title: 'Peso de la regularización',
      short: 'Cuánto pesa el término KL frente a la reconstrucción. Es el mando que equilibra "reconstruir fiel" contra "espacio latente ordenado".',
      what: 'La pérdida es <b>reconstrucción + β · KL</b>. El KL empuja a que el código de cada observación se parezca a una gaussiana estándar N(0, I): centrado, de escala 1 y sin huecos. Con β = 1 es el VAE original (' + cite('kingma2014', 'Kingma &amp; Welling, 2014') + '); con β > 1 es el <b>β-VAE</b> (' + cite('higgins2017', 'Higgins et al., 2017') + ').',
      controls: 'Controla el <b>compromiso</b> entre fidelidad y orden del espacio latente. Es el hiperparámetro más característico del VAE y el que más fácilmente rompe el modelo.',
      upLabel: 'Si lo subes', downLabel: 'Si lo bajas',
      up: 'Latente más ordenado y a veces con factores más <b>separados</b> (<span data-term="desenredo">desenredo</span>); peor reconstrucción. Pasado cierto punto se produce el <b>colapso posterior</b>: al modelo le sale más barato ignorar el código por completo (KL = 0) y reconstruir siempre la media. Medido aquí: con β ≥ 4 en «Punto móvil» quedan <b>0 de 2</b> dimensiones activas.',
      down: 'Mejor reconstrucción. Con β = 0 ya no es un VAE sino un <b>autoencoder</b> normal: el latente se desparrama (KL 68 frente a 2.6 con β = 1), quedan huecos y <b>muestrear del prior deja de producir datos válidos</b>.',
      typical: '1 para empezar. Súbelo (2–6) solo si buscas factores separados o un espacio más regular, y <b>vigila el KL</b>: si cae a cero, te pasaste.',
      tune: { level: 'high', label: 'Prioridad alta' },
      real: '<b>Ejemplo:</b> un equipo que quiere generar caras nuevas sube β para que el espacio no tenga huecos; uno que quiere detectar defectos sutiles lo baja, porque necesita que la reconstrucción sea fiel.',
      watch: 'Mira el <b>KL por dimensión</b> (¿se apagan?), el <b>Espacio latente</b> y el <b>Mapa <span data-term="generativo">generativo</span></b>.',
      target: 'card-kl',
      live: (s, m, ev) => `β = <b>${s.beta}</b> → reconstrucción ${ev.rec.toFixed(1)} · KL ${ev.kl.toFixed(2)} nats. ${ev.kl < 0.05 ? '<b>⚠ Colapso posterior:</b> el modelo está ignorando el código latente.' : ev.active + ' de ' + m.L + ' dimensiones activas.'}`,
    },
    {
      id: 'hidden', code: 'hidden_units', symbol: 'H', title: 'Capacidad de la red',
      short: 'Cuántas neuronas tiene la capa oculta del codificador y del decodificador.',
      what: 'Encoder y decoder son redes pequeñas: <code>64 → H → L</code> y <code>L → H → 64</code>, con activación tanh. H fija cuántos parámetros tiene el modelo.',
      controls: 'Controla la <b>flexibilidad</b> de las funciones que pueden aprender. Con H = 0 el modelo es lineal (algo parecido a un <span data-term="pca">PCA</span> probabilístico) y no puede capturar relaciones curvas.',
      upLabel: 'Si la subes', downLabel: 'Si la bajas',
      up: 'Mejor reconstrucción con rendimientos decrecientes y más costo por <span data-term="epoca">época</span>. Medido aquí: de H = 16 a H = 64 la reconstrucción pasa de 9.3 a 7.5 mientras el tiempo por época se triplica.',
      down: 'Más rápido, pero si la red es demasiado pequeña <b>no puede usar el código</b> aunque quiera: medido aquí, con H = 4 el KL cae a 0 (colapso por falta de capacidad, no por β).',
      typical: '16–64 para estos datos. En datos reales, capas mayores y más profundas; el criterio es el mismo: subir hasta que la curva de reconstrucción se aplane.',
      tune: { level: 'medium', label: 'Prioridad media' },
      real: '<b>Ejemplo:</b> en imágenes de verdad se usan capas convolucionales en lugar de densas; el razonamiento sobre la capacidad es idéntico.',
      watch: 'Mira la <b>Curva de entrenamiento</b> y el nº de parámetros de la ficha del modelo.',
      target: 'card-curve',
      live: (s, m) => `H = <b>${m.H || 'sin capa oculta (lineal)'}</b> → <b>${m.nParams().toLocaleString('es')}</b> parámetros entrenables.`,
    },
    {
      id: 'lr', code: 'learning_rate', symbol: 'η', title: 'Tasa de aprendizaje',
      short: 'Cuánto se mueven los pesos en cada paso del optimizador (aquí, Adam).',
      what: 'En cada lote se calcula el gradiente de la pérdida y se da un paso en esa dirección. η es el tamaño de ese paso.',
      controls: 'Controla la <b>velocidad y la estabilidad</b> del entrenamiento. No cambia qué puede aprender el modelo, sino si consigue llegar.',
      upLabel: 'Si la subes', downLabel: 'Si la bajas',
      up: 'Converge antes… hasta que se pasa y el entrenamiento se vuelve inestable o diverge. Medido aquí: con η = 0.3 la pérdida sube a 60 (frente a 10.7 con η = 0.01).',
      down: 'Más estable pero mucho más lento: puede que al terminar las épocas ni siquiera haya empezado a aprender. Medido aquí: con η = 0.0003 la reconstrucción se queda en 16.5, igual que un modelo sin entrenar.',
      typical: '0.01–0.03 en este laboratorio; 1e-3 o 3e-4 es lo habitual con Adam en redes grandes. Es el hiperparámetro <b>más importante</b> en casi cualquier red neuronal.',
      tune: { level: 'high', label: 'Prioridad alta' },
      real: '<b>Ejemplo:</b> es el primer sospechoso cuando una red "no aprende". Antes de cambiar la arquitectura, barre la tasa de aprendizaje en escala logarítmica.',
      watch: 'Mira la <b>Curva de entrenamiento</b>: si sube o serpentea, es demasiado alta; si baja apenas, demasiado baja.',
      target: 'card-curve',
      live: (s) => `η = <b>${s.lr}</b>. Barre en escala logarítmica (×3 cada vez), no sumando.`,
    },
    {
      id: 'warmup', code: 'kl_warmup', symbol: '↗', title: 'Recocido del KL (warm-up)',
      short: 'Durante las primeras épocas el peso del KL sube poco a poco desde 0 hasta β, en lugar de aplicarse de golpe.',
      what: 'Truco clásico contra el colapso posterior (' + cite('bowman2016', 'Bowman et al., 2016') + '; ' + cite('sonderby2016', 'Sønderby et al., 2016') + '): el modelo primero aprende a usar el código como un autoencoder y solo después se le exige que lo ordene.',
      controls: 'Controla <b>cuándo</b> aparece la presión del KL, no cuánta. Es una decisión de <i>optimización</i>, no de objetivo: el modelo final optimiza la misma pérdida.',
      upLabel: 'Con más épocas de recocido', downLabel: 'Sin recocido (0)',
      up: 'Más margen para que el código se vuelva útil antes de la presión. Medido aquí: con β = 4 el modelo pasa de <b>0 de 2</b> dimensiones activas (sin recocido) a <b>2 de 2</b> con 25 épocas de recocido.',
      down: 'La presión completa desde la primera época. Con β pequeño da igual; con β grande es la receta del colapso.',
      typical: '10–30 % de las épocas totales. Si tu KL colapsa, prueba esto <b>antes</b> de bajar β.',
      tune: { level: 'low', label: 'Prioridad baja (pero salva modelos)' },
      real: '<b>Ejemplo:</b> en modelos de texto con decodificadores potentes el colapso es casi la regla, y el recocido es una técnica estándar (' + cite('bowman2016', 'Bowman et al., 2016') + ').',
      watch: 'Mira la línea vertical en la <b>Curva de entrenamiento</b> y el <b>KL por dimensión</b>.',
      target: 'card-curve',
      live: (s) => (s.warmup > 0 ? `El peso del KL sube de 0 a ${s.beta} durante las primeras <b>${s.warmup}</b> épocas.` : 'Sin recocido: el KL pesa β desde la primera época.'),
    },
    {
      id: 'epochs', code: 'epochs', symbol: 'E', title: 'Épocas de entrenamiento',
      short: 'Cuántas veces el modelo recorre todo el conjunto de datos; cada pasada completa es una <span data-term="epoca">época</span>.',
      what: 'Cada época actualiza los pesos con todos los lotes. Aquí puedes entrenar por tramos y ver la evolución.',
      controls: 'Controla <b>cuánto</b> se entrena, no cómo. Junto con la tasa de aprendizaje determina si el modelo converge.',
      upLabel: 'Si las subes', downLabel: 'Si las bajas',
      up: 'La pérdida baja hasta aplanarse (el <span data-term="plateau">plateau</span>). Pasado ese punto solo gastas tiempo, y en datos reales puedes empezar a <span data-term="sobreajuste">sobreajustar</span> (la pérdida de validación sube mientras la de entrenamiento sigue bajando).',
      down: 'Modelo a medio entrenar: reconstrucciones borrosas y latente sin estructura. Es el error más fácil de confundir con "el modelo no sirve".',
      typical: 'Entrena hasta que la curva se aplane y usa parada temprana con un conjunto de validación. No es un hiperparámetro a optimizar por sí mismo.',
      tune: { level: 'no', label: 'Fíjalo por la curva' },
      real: '<b>Ejemplo:</b> en producción se usa <i>early stopping</i>: se guarda el modelo con mejor pérdida de validación y se detiene si no mejora en N épocas.',
      watch: 'Mira la <b>Curva de entrenamiento</b>: ¿sigue bajando o ya está plana?',
      target: 'card-curve',
      live: (s, m) => `<b>${m.epoch}</b> épocas entrenadas${m.hist.length > 6 ? ` · últimas 5 épocas: la reconstrucción cambió ${(m.hist[m.hist.length - 6].rec - m.hist[m.hist.length - 1].rec).toFixed(2)} nats` : ''}.`,
    },
    {
      id: 'batch', code: 'batch_size', symbol: 'B', title: 'Tamaño de lote',
      short: 'Cuántas observaciones se promedian antes de cada actualización de los pesos.',
      what: 'Con lote 1 se actualiza tras cada observación (muy ruidoso); con lote = N se usa todo el conjunto en cada paso (muy suave pero pocos pasos por época).',
      controls: 'Controla el <b>ruido del gradiente</b> y la velocidad por época. Interactúa con la tasa de aprendizaje: lotes más grandes suelen tolerar (y necesitar) η mayor.',
      upLabel: 'Con lotes grandes', downLabel: 'Con lotes pequeños',
      up: 'Gradientes más estables y mejor aprovechamiento del hardware, pero menos actualizaciones por época: puede necesitar más épocas.',
      down: 'Más actualizaciones y algo de ruido que a veces ayuda a escapar de malos mínimos, pero entrenamiento más lento por época y más inestable.',
      typical: '16–128. Ajústalo junto con la tasa de aprendizaje, no por separado.',
      tune: { level: 'low', label: 'Prioridad baja' },
      real: '<b>Ejemplo:</b> en la práctica lo fija la memoria de la GPU más que la calidad del modelo.',
      watch: 'Mira la <b>Curva de entrenamiento</b>: con lotes pequeños es más irregular.',
      target: 'card-curve',
      live: (s) => `Lote = <b>${s.batch}</b> → ${Math.ceil(s.n / s.batch)} actualizaciones por época.`,
    },
    {
      id: 'seed', code: 'random_state', symbol: '#', title: 'Semilla aleatoria',
      short: 'Fija la inicialización de los pesos, el barajado de los lotes y el ruido del muestreo latente.',
      what: 'Un VAE tiene tres fuentes de azar: los pesos iniciales, el orden de los datos y el ruido ε de la reparametrización. La semilla las fija todas.',
      controls: 'Controla la <b>reproducibilidad</b>. Si cambiar la semilla cambia mucho el resultado, el modelo tiene demasiada varianza.',
      upLabel: 'Buscar la "mejor" semilla', downLabel: 'Buena práctica',
      up: 'Es sobreajustar al azar: esa ventaja no existe en datos nuevos.',
      down: 'Fíjala para reproducir y varíala (3–5 valores) para reportar media ± desviación. Medido aquí: el ELBO varía 10.69 ± 0.13 entre 5 semillas.',
      typical: 'Cualquier entero fijo. Nunca la optimices.',
      tune: { level: 'no', label: 'Nunca optimizar' },
      real: '<b>Ejemplo:</b> dos entrenamientos con distinta semilla pueden dar espacios latentes <b>rotados o reflejados</b> entre sí y ser igual de buenos; por eso el latente no es comparable entre ejecuciones.',
      watch: 'Reentrena con otra semilla y compara el <b>Espacio latente</b>: la forma se conserva, la orientación no.',
      target: 'card-latent',
      live: (s) => `Semilla = <b>${s.seed}</b>. Cambiarla reinicia el entrenamiento desde cero.`,
    },
  ];

  /* Estándar compartido del capítulo "Optimizar" (assets/js/core/tuning-doc.js). */
  ML.TUNING_DOC = {
    notTune: [
      { code: 'epochs', kind: 'fix',
        why: 'No es una palanca de calidad sino de <b>cuánto entrenas</b>. Entrenar de más gasta tiempo y, con datos reales, empieza a sobreajustar.',
        instead: 'entrena hasta que la curva de validación se aplane y usa <b>parada temprana</b> guardando el mejor modelo. Fija un máximo generoso y deja que la parada decida.' },
      { code: 'random_state', kind: 'repro',
        why: 'Elegir la semilla que da la mejor métrica es <b>sobreajustar al azar</b>. En un VAE además cambia la orientación del espacio latente, que es arbitraria.',
        instead: 'fíjala para reproducir y <b>varíala</b> (3–5 valores) para reportar media ± desviación.' },
      { code: 'umbral de anomalía', kind: 'decide',
        why: 'El error de reconstrucción a partir del cual algo se declara anómalo <b>no mejora el modelo</b>: solo decide cuántas alertas emites. Es el mismo papel que <code>contamination</code> en Isolation Forest.',
        instead: 'fíjalo por la <b>capacidad de revisión</b> del equipo y el costo relativo de los errores, usando la curva Precisión–Recall.' },
      { code: 'kl_warmup', kind: 'fix',
        why: 'Es una ayuda de <b>optimización</b>, no un objetivo: cambia cómo llegas, no adónde quieres llegar. Buscar su valor óptimo con una búsqueda fina rara vez compensa.',
        instead: 'actívalo (10–30 % de las épocas) si ves colapso posterior y déjalo fijo. Si con recocido sigue colapsando, el problema es β o la arquitectura.' },
    ],
  };

  /* =====================================================================
   *  Métricas
   * ===================================================================== */
  ML.VAE_METRICS = {
    rec: { label: 'Reconstrucción', group: 'fit', dir: -1,
      tip: '<span class="tip-h">Error de reconstrucción (nats)</span>Cuánta información se pierde al comprimir y reconstruir, en entropía cruzada binaria sumada sobre los 64 píxeles. <b>Menor es mejor.</b><span class="tip-sub">No necesita etiquetas. Es una de las dos mitades de la pérdida.</span>' },
    kl: { label: 'KL', group: 'fit', dir: -1,
      tip: '<span class="tip-h">Divergencia KL (nats)</span>Cuánto se aleja el código de cada observación del prior N(0, I). Mide la <b>información que el latente transporta</b>: KL = 0 significa que el código no dice nada.<span class="tip-sub">No necesita etiquetas. Ni muy alto (latente desordenado) ni cero (colapso).</span>' },
    elbo: { label: 'Pérdida (−ELBO)', group: 'fit', dir: -1,
      tip: '<span class="tip-h">−ELBO = reconstrucción + KL</span>Cota inferior de la evidencia, con signo cambiado: es lo que el entrenamiento minimiza cuando β = 1. <b>Menor es mejor.</b><span class="tip-sub">No necesita etiquetas. Con β ≠ 1 el modelo optimiza otra cosa, así que compara con cuidado.</span>' },
    active: { label: 'Dimensiones activas', group: 'fit',
      tip: '<span class="tip-h">Dimensiones latentes activas</span>Cuántas dimensiones tienen KL > 0.01 nats, es decir, realmente transportan información. Las demás están <b>apagadas</b>.<span class="tip-sub">No necesita etiquetas. Es el diagnóstico central del colapso posterior.</span>' },
    dis: { label: 'Alineación con factores', group: 'fact',
      tip: '<span class="tip-h">Alineación con los factores reales</span>Para cada factor que generó los datos (posición, tamaño…), la mayor correlación de rangos con alguna dimensión latente, promediada. Cerca de 1 = cada factor vive en su propia dimensión (<b>desenredo</b>).<span class="tip-sub">Solo se puede calcular porque estos datos son sintéticos y conocemos los factores. Con datos reales no existe.</span>' },
    auc: { label: 'ROC-AUC (anomalías)', group: 'anom',
      tip: '<span class="tip-h">ROC-AUC del error de reconstrucción</span>Probabilidad de que una anomalía real tenga mayor error de reconstrucción que una observación normal.<span class="tip-sub">Necesita etiquetas. Solo aparece si el conjunto tiene anomalías.</span>' },
    ap: { label: 'AP (anomalías)', group: 'anom',
      tip: '<span class="tip-h">Precisión media</span>Área bajo la curva Precisión–Recall usando el error de reconstrucción como score. Más exigente que el ROC-AUC cuando las anomalías son raras.<span class="tip-sub">Necesita etiquetas.</span>' },
  };

  /* =====================================================================
   *  Escenarios guiados
   * ===================================================================== */
  ML.VAE_LESSONS = [
    { id: 'first', title: 'Primer entrenamiento: ver cómo aprende', tag: 'épocas',
      setup: { dataset: 'dot', latent: 2, beta: 1, hidden: 24, lr: 0.01, warmup: 0, batch: 16, seed: 1, epochs: 0 }, active: 'epochs', target: 'card-curve',
      steps: ['Pulsa <b>+20 épocas</b> tres veces y observa las reconstrucciones.', 'Fíjate en la curva: la reconstrucción baja y el KL sube hasta estabilizarse.'],
      look: 'Al principio las reconstrucciones son manchas grises (el modelo predice la media). Según entrena, aparecen los discos en la posición correcta y el <b>espacio latente</b> se despliega. Cuando la curva se aplana, más épocas ya no aportan.' },
    { id: 'beta-collapse', title: 'β alto: el colapso posterior', tag: 'beta',
      setup: { dataset: 'dot', latent: 2, beta: 1, hidden: 24, lr: 0.01, warmup: 0, batch: 16, seed: 1, epochs: 60 }, active: 'beta', target: 'card-kl',
      steps: ['Con β = 1, mira el KL por dimensión: dos barras activas.', 'Sube β a 2, luego a 4 y a 8, reentrenando cada vez.'],
      look: 'Con β = 4 las barras caen a cero: <b>colapso posterior</b>. El modelo decide que le sale más barato ignorar el código y reconstruir siempre lo mismo; el espacio latente se convierte en una bola sin estructura y las reconstrucciones, en una mancha promedio. Es el fallo más típico del VAE (' + cite('bowman2016', 'Bowman et al., 2016') + ').' },
    { id: 'rescue', title: 'Rescatar el colapso con recocido', tag: 'kl_warmup',
      setup: { dataset: 'dot', latent: 2, beta: 4, hidden: 24, lr: 0.01, warmup: 0, batch: 16, seed: 1, epochs: 60 }, active: 'warmup', target: 'card-kl',
      steps: ['Empieza con β = 4 sin recocido: el KL está en cero.', 'Pon <b>recocido = 25</b> épocas y reentrena.'],
      look: 'Con el mismo β, el modelo pasa de <b>0 a 2 dimensiones activas</b>: dejar que primero aprenda a usar el código y solo después exigirle orden cambia por completo el resultado. Con β = 8 ni el recocido lo salva: ahí el problema es β.' },
    { id: 'beta0', title: 'β = 0: esto ya no es un VAE', tag: 'beta',
      setup: { dataset: 'dot', latent: 2, beta: 0, hidden: 24, lr: 0.01, warmup: 0, batch: 16, seed: 1, epochs: 60 }, active: 'beta', target: 'card-gen',
      steps: ['Con β = 0 mira el espacio latente y el mapa generativo.', 'Vuelve a β = 1 y compara.'],
      look: 'Con β = 0 la reconstrucción es buena, pero el latente se <b>desparrama</b> (KL ≈ 68 frente a 2.6) y deja huecos enormes. En el <b>mapa generativo</b>, que decodifica puntos del prior, salen manchas sin sentido: el modelo ya no es generativo. Eso es un autoencoder normal, no un VAE.' },
    { id: 'toosmall', title: 'Latente demasiado pequeño', tag: 'latent_dim',
      setup: { dataset: 'dotsize', latent: 2, beta: 1, hidden: 24, lr: 0.01, warmup: 0, batch: 16, seed: 1, epochs: 80 }, active: 'latent', target: 'card-fact',
      steps: ['Estos datos tienen <b>3 factores</b>: posición x, posición y y tamaño, pero el latente es de 2.', 'Mira la alineación por factor. Luego sube latent_dim a 3 y reentrena.'],
      look: 'Con L = 2 el modelo codifica bien la posición y <b>abandona el tamaño</b> (correlación 0.08): no le caben tres cosas en dos números. Con L = 3 el tamaño empieza a aparecer. Cuando un factor importante no se codifica, el modelo es ciego a él, y eso se nota luego en la detección de anomalías.' },
    { id: 'lr', title: 'La tasa de aprendizaje manda', tag: 'learning_rate',
      setup: { dataset: 'dot', latent: 2, beta: 1, hidden: 24, lr: 0.0003, warmup: 0, batch: 16, seed: 1, epochs: 60 }, active: 'lr', target: 'card-curve',
      steps: ['Con η = 0.0003 el modelo apenas aprende en 60 épocas.', 'Prueba 0.01 y luego 0.3, reentrenando cada vez.'],
      look: 'Con η demasiado baja la curva casi no baja: parece que el modelo no sirve, pero solo le falta paso. Con η = 0.3 la pérdida <b>explota</b>. El intervalo útil abarca dos órdenes de magnitud: por eso se barre en escala logarítmica.' },
    { id: 'anom', title: 'Detectar anomalías con el error de reconstrucción', tag: 'aplicación',
      setup: { dataset: 'dotsize', latent: 3, beta: 1, hidden: 24, lr: 0.01, warmup: 0, batch: 16, seed: 1, epochs: 70, anomalies: 0.06 }, active: 'beta', target: 'card-anom',
      steps: ['El conjunto tiene un 6 % de anomalías (barras donde siempre hubo discos).', 'Entrena y mira el histograma de errores. Luego sube β a 4 y reentrena.'],
      look: 'El VAE aprende a reconstruir lo normal; lo que nunca vio le sale mal, y ese error es el score de anomalía. Con β = 4 el modelo colapsa y la precisión media se hunde (medido aquí: AP de 1.00 a 0.27). <b>Un modelo colapsado sigue dando scores</b>, solo que malos: por eso hay que vigilar el KL, no solo la pérdida.' },
  ];

  /* =====================================================================
   *  Casos reales
   * ===================================================================== */
  ML.CASES = ML.VAE_CASES = [
    { id: 'industria', name: 'Inspección visual de piezas', kicker: 'Industria',
      ctx: 'Una cámara fotografía cada pieza en la línea. Hay millones de piezas buenas y muy pocas defectuosas, y los defectos nuevos no se parecen a los antiguos.',
      goal: 'Marcar piezas con aspecto <b>inusual</b> para inspección humana, sin una lista previa de defectos.',
      why: 'Se entrena solo con piezas buenas: el VAE aprende a reconstruirlas. Un defecto que nunca vio se reconstruye mal, y el <b>mapa de error por píxel</b> señala dónde está.',
      keys: [['latent_dim', 'Suficiente para las variaciones legítimas (iluminación, posición). Si te quedas corto, el modelo marcará piezas buenas como raras.'],
        ['beta (β)', 'Bajo o igual a 1: aquí interesa la <b>fidelidad</b> de la reconstrucción, no un latente bonito. Un modelo colapsado reconstruye la media y pierde los defectos sutiles.'],
        ['umbral', 'Fíjalo por cuántas piezas puede revisar el inspector por turno.']],
      measure: 'AP y Precisión@k sobre defectos confirmados; además, revisión visual de los mapas de error.',
      trap: 'Si en el entrenamiento se colaron piezas defectuosas, el VAE aprende a reconstruirlas y deja de detectarlas.',
      sim: { lesson: 'anom', label: 'Simular detección' } },
    { id: 'salud', name: 'Señales fisiológicas', kicker: 'Salud',
      ctx: 'Latidos o segmentos de ECG de pacientes monitorizados. Los ritmos anómalos son raros y heterogéneos.',
      goal: 'Levantar segmentos atípicos para revisión de un especialista.',
      why: 'Se entrena con ritmo normal y se mide el error de reconstrucción. Además, el espacio latente agrupa morfologías parecidas, lo que ayuda a explorar.',
      keys: [['latent_dim', 'Pequeño y luego creciente: cuenta cuántas dimensiones quedan <b>activas</b>; eso sugiere cuántos modos de variación reales hay.'],
        ['beta (β)', 'Vigila el KL en cada reentreno. Un colapso silencioso convierte el detector en un generador de ruido con apariencia normal.'],
        ['random_state', 'Fíjala: la trazabilidad de por qué se marcó un segmento es un requisito clínico.']],
      measure: 'Recall con precisión mínima aceptable, revisión clínica de una muestra y estabilidad entre reentrenos.',
      trap: 'El "normal" depende del paciente y de la actividad. Un modelo global marcará como anómalo a cualquiera fuera de la media.',
      sim: { lesson: 'anom', label: 'Simular detección' } },
    { id: 'datos', name: 'Datos sintéticos y aumento', kicker: 'Ciencia de datos',
      ctx: 'Necesitas más ejemplos de una clase rara, o compartir datos sin exponer los originales.',
      goal: 'Generar observaciones <b>nuevas y plausibles</b> muestreando del prior y decodificando.',
      why: 'A diferencia de un autoencoder normal, el VAE fuerza a que el latente se parezca a una gaussiana: se puede muestrear de ella y obtener datos válidos.',
      keys: [['beta (β)', 'Aquí sí conviene β ≥ 1: un latente con huecos produce muestras corruptas. Es justo el compromiso opuesto al de la inspección de piezas.'],
        ['latent_dim', 'Demasiado grande deja dimensiones apagadas y muestras poco variadas.']],
      measure: 'Inspección visual del <b>mapa generativo</b>, diversidad de las muestras y utilidad real (¿mejora el modelo entrenado con ellas?).',
      trap: 'Las muestras de un VAE tienden a ser <b>borrosas</b>: es una limitación conocida del objetivo, no un error de configuración.',
      sim: { lesson: 'beta0', label: 'Ver β = 0 vs 1' } },
    { id: 'repr', name: 'Compresión y representación', kicker: 'Ingeniería de datos',
      ctx: 'Tienes cientos de sensores correlacionados y quieres una representación compacta para alimentar otros modelos o para visualizar.',
      goal: 'Reducir a unas pocas variables que conserven la información útil.',
      why: 'El VAE es una reducción de dimensionalidad <b>no lineal</b> y probabilística. Con la red lineal (H = 0) se parece a un PCA probabilístico; con capas ocultas captura relaciones curvas que el PCA no.',
      keys: [['latent_dim', 'El número de dimensiones <b>activas</b> es una estimación de la complejidad real de tus datos.'],
        ['hidden_units', 'Con H = 0 comparas contra el equivalente lineal: si no mejora, tus datos no necesitan un modelo no lineal.']],
      measure: 'Reconstrucción frente al número de dimensiones activas, y el rendimiento del modelo posterior que use esa representación.',
      trap: 'Las dimensiones latentes <b>no son interpretables por defecto</b> y cambian de orientación con cada semilla: no las trates como variables con nombre sin comprobarlo.',
      sim: { lesson: 'toosmall', label: 'Ver latente insuficiente' } },
  ];

  /* =====================================================================
   *  Preguntas de comprobación
   * ===================================================================== */
  ML.CHECKS = ML.VAE_CHECKS = {
    ch1: [
      { q: '¿Por qué el encoder produce una <i>distribución</i> (μ y σ) en vez de un solo punto?', a: 'Para que el espacio latente sea <b>continuo y sin huecos</b>: cada observación ocupa una región difusa, no un punto aislado. Así, puntos cercanos decodifican a cosas parecidas y se puede muestrear del espacio para generar datos nuevos.' },
      { q: '¿Qué hace el truco de reparametrización y por qué es necesario?', a: 'Escribe z = μ + σ·ε con ε ~ N(0,1) fijo en cada paso. El azar queda <b>fuera</b> del camino de los parámetros, así que se puede derivar respecto a μ y σ y entrenar con retropropagación. Sin él no habría gradiente a través del muestreo.' },
    ],
    ch2: [
      { q: 'Entrenas y el KL está exactamente en 0. ¿Qué pasó y qué pruebas primero?', a: '<b>Colapso posterior</b>: el modelo ignora el código. Prueba, en este orden: activar el <b>recocido</b> del KL, bajar β, y comprobar que la red tiene capacidad suficiente (con H muy pequeña también colapsa).' },
      { q: 'Subes latent_dim de 4 a 16 y la reconstrucción no mejora. ¿Desperdiciaste capacidad?', a: 'No: el VAE <b>apaga solo</b> las dimensiones que no necesita (su KL cae a ~0). Mira cuántas quedan activas; ese número te dice cuántas pide el problema. Lo único que pierdes es tiempo de cómputo.' },
    ],
    ch3: [
      { q: 'Un modelo tiene mejor pérdida total que otro. ¿Es mejor?', a: 'Depende de para qué, y solo son comparables <b>con el mismo β</b>. Un modelo puede ganar bajando el KL a costa de ignorar el latente: mejor pérdida, peor representación. Mira siempre las dos partes por separado y las dimensiones activas.' },
      { q: 'No tienes etiquetas ni conoces los factores. ¿Cómo evalúas?', a: 'Reconstrucción y KL por separado, <b>dimensiones activas</b>, inspección visual de reconstrucciones y del mapa generativo, estabilidad entre semillas y, si hay un objetivo posterior, el rendimiento de ese objetivo.' },
    ],
    ch4: [
      { q: '¿Por qué β y latent_dim se deben ajustar juntos y no uno por uno?', a: 'Porque <b>interactúan</b>: β grande apaga dimensiones, así que subir latent_dim con β alto no sirve de nada. La cuadrícula β × latent_dim muestra que el óptimo está en una región, no en un valor de cada uno por separado.' },
      { q: 'En un VAE, ¿qué papel juega la tasa de aprendizaje frente a los hiperparámetros del modelo?', a: 'Es de <b>optimización</b>: no cambia qué puede representar el modelo, sino si consigue llegar. Conviene ajustarla primero, porque un η mal puesto hace que cualquier otra configuración parezca mala.' },
    ],
    ch5: [
      { q: 'Inspección de piezas y generación de datos sintéticos tiran de β en direcciones opuestas. ¿Por qué?', a: 'La inspección necesita <b>fidelidad</b> (detectar defectos sutiles) → β bajo. La generación necesita un latente <b>sin huecos</b> para muestrear → β ≥ 1. El mismo hiperparámetro, objetivos contrarios: por eso no existe un valor universal.' },
    ],
    ch6: [
      { q: 'Vas a poner un VAE en producción para detectar anomalías. ¿Qué tres cosas monitoreas?', a: '1) El <b>KL</b> en cada reentreno (un colapso silencioso rompe el detector sin que la pérdida lo delate); 2) la <b>distribución del error de reconstrucción</b> (deriva de datos); 3) la <b>estabilidad</b> de la lista de alertas y la tasa de confirmación de los revisores.' },
    ],
  };

  /* =====================================================================
   *  Glosario
   * ===================================================================== */
  ML.GLOSSARY = {
    vae: ['Autoencoder variacional (VAE)', 'Red que comprime cada observación en una <b>distribución</b> sobre un espacio latente y la reconstruye desde una muestra de esa distribución. Aprende sin etiquetas y puede generar datos nuevos.'],
    autoencoder: ['Autoencoder', 'Red que comprime y reconstruye. En su versión clásica el código es un punto fijo; el VAE lo convierte en una distribución y añade el término KL.'],
    latente: ['Espacio latente', 'El espacio comprimido donde vive el código z de cada observación. Sus dimensiones no tienen nombre a priori.'],
    encoder: ['Encoder (codificador)', 'La mitad que va de la observación al código: produce la media μ y la varianza σ² de la distribución latente.'],
    decoder: ['Decoder (decodificador)', 'La mitad que va del código a la reconstrucción. Es también el <b>generador</b>: decodificar un punto cualquiera produce un dato nuevo.'],
    reparam: ['Truco de reparametrización', 'Escribir z = μ + σ·ε con ε ~ N(0,1). Saca el azar del camino de los parámetros y permite entrenar con retropropagación (Kingma &amp; Welling, 2014).'],
    kl: ['Divergencia KL', 'Mide cuánto se aleja la distribución del código, q(z|x), del prior N(0, I). En un VAE es la "cuota" que se paga por usar información en el latente. Se mide en <b>nats</b>.'],
    elbo: ['ELBO', '<i>Evidence Lower Bound</i>: cota inferior de la verosimilitud de los datos. Maximizarla equivale a minimizar reconstrucción + KL. Aquí se muestra con el signo cambiado, como pérdida.'],
    nats: ['Nats', 'Unidad de información, como los bits pero con logaritmo natural (1 nat ≈ 1.44 bits). Reconstrucción y KL se miden en nats.'],
    prior: ['Prior', 'La distribución que se le pide al espacio latente, aquí una gaussiana estándar N(0, I). Es de donde se muestrea para generar datos nuevos.'],
    posterior: ['Posterior aproximado', 'La distribución q(z|x) que el encoder produce para una observación concreta.'],
    colapso: ['Colapso posterior', 'El modelo deja de usar el código latente: q(z|x) se vuelve igual al prior y el KL cae a 0. La reconstrucción pasa a ser siempre la media. Se combate con recocido del KL o bajando β (Bowman et al., 2016).'],
    recocido: ['Recocido del KL (warm-up)', 'Subir el peso del KL poco a poco desde 0 durante las primeras épocas, para que el modelo aprenda a usar el código antes de que se le exija ordenarlo.'],
    desenredo: ['Desenredo (disentanglement)', 'Que cada dimensión latente corresponda a un factor interpretable e independiente de los datos. β > 1 lo favorece a costa de la reconstrucción (Higgins et al., 2017).'],
    beta: ['β (beta)', 'Peso del término KL en la pérdida. β = 1 es el VAE original; β > 1 es el β-VAE.'],
    reconstruccion: ['Error de reconstrucción', 'Cuánto se diferencia la salida de la entrada. Aquí, entropía cruzada binaria sumada sobre los 64 píxeles, en nats. Sirve además como score de anomalía.'],
    epoca: ['Época', 'Una pasada completa por todo el conjunto de entrenamiento.'],
    lote: ['Lote (batch)', 'Grupo de observaciones que se promedian antes de cada actualización de los pesos.'],
    adam: ['Adam', 'Optimizador que adapta el tamaño del paso por parámetro usando medias móviles del gradiente. El más habitual en aprendizaje profundo.'],
    sobreajuste: ['Sobreajuste', 'El modelo memoriza el conjunto de entrenamiento y empeora en datos nuevos: la pérdida de entrenamiento baja mientras la de validación sube.'],
    plateau: ['Plateau', 'Zona en la que seguir entrenando (o subir un valor) ya no mejora la métrica de forma medible.'],
    hiperparametro: ['Hiperparámetro', 'Ajuste del algoritmo que tú eliges antes de entrenar. Se distingue de los <i>parámetros</i> (los pesos), que el modelo aprende.'],
    generativo: ['Modelo generativo', 'Modelo del que se pueden extraer datos nuevos, no solo clasificar los existentes. El VAE lo es; un autoencoder normal, no.'],
    scoreanom: ['Error de reconstrucción como score', 'Usar lo mal que el modelo reconstruye una observación como medida de rareza. Es la forma habitual de convertir un VAE en detector de anomalías.'],
    umbral: ['Umbral de decisión', 'Valor del score a partir del cual una observación se declara anómala. Fija cuántas alertas emites; no cambia el modelo.'],
    pr: ['Curva Precisión–Recall', 'Precisión frente a recall para todos los umbrales posibles. Más informativa que la ROC cuando las anomalías son raras.'],
    patk: ['Precisión@k', 'De las k observaciones con mayor score, qué fracción son anomalías reales. Refleja la capacidad de revisión real de un equipo.'],
    grid: ['Grid search', 'Búsqueda exhaustiva sobre una malla de valores. Su costo crece exponencialmente con el número de hiperparámetros.'],
    random: ['Random search', 'Búsqueda que muestrea configuraciones al azar. Con el mismo presupuesto prueba más valores distintos de cada hiperparámetro (Bergstra y Bengio, 2012).'],
    bayes: ['Optimización bayesiana', 'Construye un modelo de cómo los hiperparámetros afectan la métrica y elige la siguiente prueba más prometedora. Útil cuando cada evaluación es cara.'],
    halving: ['Successive Halving / Hyperband', 'Entrena muchas configuraciones con poco recurso (pocas épocas) y solo promueve las mejores. Reduce mucho el costo de la búsqueda.'],
    pca: ['PCA', 'Análisis de componentes principales: la reducción de dimensionalidad lineal clásica. Un VAE lineal (sin capa oculta) se le parece mucho.'],
    varianza: ['Varianza (del modelo)', 'Cuánto cambia el resultado solo por el azar (semilla, inicialización, orden de los datos).'],
    aucroc: ['ROC-AUC', 'Probabilidad de que una anomalía real puntúe más alto que una normal. Necesita etiquetas.'],
    ap: ['Precisión media (AP)', 'Área bajo la curva Precisión–Recall. Mejor que el ROC-AUC cuando las anomalías son raras.'],
  };

  /* =====================================================================
   *  Resúmenes
   * ===================================================================== */
  ML.TAKEAWAYS = ML.VAE_TAKEAWAYS = {
    ch1: ['El VAE comprime cada observación en una <b>distribución</b> sobre un espacio latente y la reconstruye desde una muestra.',
      'Su pérdida tiene dos partes en tensión: <b>reconstrucción</b> (ser fiel) y <b>KL</b> (que el latente se parezca al prior).',
      'El truco de reparametrización permite entrenar a través del muestreo aleatorio.'],
    ch2: ['<b>β</b> y <b>latent_dim</b> deciden qué representa el modelo; <b>learning_rate</b>, <b>épocas</b> y <b>lote</b> deciden si llega a aprenderlo.',
      'Con β alto aparece el <b>colapso posterior</b> (KL = 0): el modelo ignora el latente y reconstruye la media.',
      'El VAE <b>apaga solo</b> las dimensiones latentes que no necesita: el número de activas te dice cuántas piden los datos.'],
    ch3: ['Reconstrucción y KL hay que mirarlos <b>por separado</b>; la pérdida total sola puede engañar, y solo es comparable con el mismo β.',
      'Las <b>dimensiones activas</b> son el diagnóstico central: sin etiquetas, es lo primero que se revisa.',
      'La alineación con factores reales solo se puede medir en datos sintéticos; en la práctica se sustituye por inspección visual y por el rendimiento posterior.'],
    ch4: ['β y latent_dim <b>interactúan</b>: hay que buscarlos juntos, no uno por uno.',
      'La tasa de aprendizaje se ajusta primero y en escala <b>logarítmica</b>: mal puesta, hace que todo lo demás parezca malo.',
      'Épocas, semilla, umbral y recocido <b>no</b> son hiperparámetros a optimizar: se fijan o se deciden.'],
    ch5: ['El mismo hiperparámetro tira en direcciones opuestas según el objetivo: β bajo para detectar defectos, β ≥ 1 para generar.',
      'Entrenar solo con datos normales es la receta habitual en detección de anomalías con VAE.',
      'Vigila el KL en producción: un colapso silencioso rompe el detector sin que la pérdida lo delate.'],
  };
})();
