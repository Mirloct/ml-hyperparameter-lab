/*
 * Catálogo de algoritmos del sitio.
 * Taxonomía en tres ejes, la que se usa para navegar y filtrar:
 *   paradigma   : ml | dl                        (Machine Learning "clásico" o Deep Learning)
 *   supervision : sup | unsup | semi | self      (según qué etiquetas necesita para ENTRENAR)
 *   family      : familia metodológica (bagging, boosting, isolation, centroid, density, ...)
 * Cada algoritmo declara además su tarea y sus hiperparámetros clave.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.MLLab = root.MLLab || {}; root.MLLab.catalog = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const PARADIGMS = {
    ml: { name: 'Machine Learning', blurb: 'Modelos que trabajan sobre variables ya definidas (tablas, señales resumidas). Rápidos, interpretables y con pocos hiperparámetros.' },
    dl: { name: 'Deep Learning', blurb: 'Redes neuronales que aprenden su propia representación de los datos. Más hiperparámetros y más sensibles a ellos.' },
  };
  const SUPERVISION = {
    sup: { name: 'Supervisado', short: 'Sup.', tip: 'Necesita <b>etiquetas</b> para cada observación de entrenamiento (la respuesta correcta).' },
    unsup: { name: 'No supervisado', short: 'No sup.', tip: 'Entrena <b>sin etiquetas</b>: busca estructura en los datos (grupos, densidad, rarezas).' },
    semi: { name: 'Semi-supervisado', short: 'Semi', tip: 'Usa <b>muchos datos sin etiquetar y unos pocos etiquetados</b>. En detección de anomalías suele significar entrenar solo con datos normales.' },
    self: { name: 'Autosupervisado', short: 'Auto', tip: 'Crea su propia señal de entrenamiento a partir de los datos (p. ej. reconstruir la entrada). No necesita etiquetas humanas.' },
  };
  const FAMILIES = {
    isolation: { name: 'Aislamiento', tip: 'Aíslan observaciones con particiones aleatorias; lo raro se separa antes.' },
    bagging: { name: 'Bagging (ensembles paralelos)', tip: 'Muchos modelos entrenados en paralelo sobre submuestras; se promedian. Reducen la <b>varianza</b>.' },
    boosting: { name: 'Boosting (ensembles secuenciales)', tip: 'Cada modelo corrige los errores del anterior. Reducen el <b>sesgo</b>, pero pueden sobreajustar.' },
    centroid: { name: 'Centroides', tip: 'Representan cada grupo por un punto central y asignan por cercanía.' },
    density: { name: 'Densidad', tip: 'Definen grupos o anomalías según qué tan poblada está la vecindad de cada punto.' },
    margin: { name: 'Margen y kernels', tip: 'Buscan la frontera que mejor separa las clases, con un margen máximo.' },
    linear: { name: 'Modelos lineales', tip: 'Combinación lineal de las variables; base de la regresión y la clasificación clásica.' },
    autoencoder: { name: 'Autoencoders', tip: 'Redes que comprimen la entrada y la reconstruyen; lo que no logran reconstruir bien es candidato a anomalía.' },
    convnet: { name: 'Redes convolucionales', tip: 'Aprovechan la estructura espacial de imágenes y señales.' },
    sequence: { name: 'Secuencias y atención', tip: 'Modelan dependencias temporales o de orden (RNN, LSTM, Transformers).' },
  };

  const ALGOS = [
    {
      id: 'isolation-forest', name: 'Isolation Forest', href: 'algorithms/isolation-forest/', status: 'live',
      paradigm: 'ml', supervision: 'unsup', family: 'isolation', task: 'Detección de anomalías',
      blurb: 'Curso completo: la idea, los 6 hiperparámetros, cómo evaluar con y sin etiquetas, optimización, casos reales y bibliografía.',
      hp: ['n_estimators', 'max_samples', 'contamination', 'max_features'],
      art: 'iforest',
    },
    {
      id: 'vae', name: 'Autoencoder Variacional (VAE)', href: 'algorithms/vae/', status: 'live',
      paradigm: 'dl', supervision: 'self', family: 'autoencoder', task: 'Representación y detección de anomalías',
      blurb: 'Comprime los datos a un espacio latente probabilístico y los reconstruye. Explora β, la dimensión latente, la capacidad de la red y el equilibrio reconstrucción–regularización.',
      hp: ['latent_dim', 'beta', 'hidden', 'learning_rate'],
      art: 'vae',
    },
    { id: 'kmeans', name: 'K-Means', status: 'soon', paradigm: 'ml', supervision: 'unsup', family: 'centroid', task: 'Clustering', blurb: 'Efecto de <code>k</code>, la inicialización y el criterio de convergencia.', hp: ['n_clusters', 'init', 'n_init'] },
    { id: 'dbscan', name: 'DBSCAN', status: 'soon', paradigm: 'ml', supervision: 'unsup', family: 'density', task: 'Clustering', blurb: 'Cómo <code>eps</code> y <code>min_samples</code> cambian los grupos y qué se considera ruido.', hp: ['eps', 'min_samples'] },
    { id: 'lof', name: 'Local Outlier Factor', status: 'soon', paradigm: 'ml', supervision: 'unsup', family: 'density', task: 'Detección de anomalías', blurb: 'Anomalías <i>locales</i>: densidad de un punto frente a la de sus vecinos.', hp: ['n_neighbors', 'contamination'] },
    { id: 'random-forest', name: 'Random Forest', status: 'soon', paradigm: 'ml', supervision: 'sup', family: 'bagging', task: 'Clasificación y regresión', blurb: 'Profundidad, número de árboles y <code>mtry</code>: el compromiso sesgo–varianza.', hp: ['n_estimators', 'max_depth', 'max_features'] },
    {
      id: 'gradient-boosting', name: 'Gradient Boosting / XGBoost', href: 'algorithms/gradient-boosting/', status: 'live',
      paradigm: 'ml', supervision: 'sup', family: 'boosting', task: 'Clasificación',
      blurb: 'Curso completo: árboles secuenciales que se corrigen entre sí, el equilibrio learning_rate–n_estimators, sobreajuste y early stopping, regularización, optimización y casos reales.',
      hp: ['learning_rate', 'n_estimators', 'max_depth', 'gamma'],
      art: 'gbm',
    },
    { id: 'svm', name: 'SVM (kernel)', status: 'soon', paradigm: 'ml', supervision: 'sup', family: 'margin', task: 'Clasificación', blurb: 'Efecto de <code>C</code> y <code>gamma</code> sobre la frontera de decisión y el margen.', hp: ['C', 'gamma', 'kernel'] },
    { id: 'one-class-svm', name: 'One-Class SVM', status: 'soon', paradigm: 'ml', supervision: 'semi', family: 'margin', task: 'Detección de anomalías', blurb: 'Entrenado solo con datos normales: <code>nu</code> y <code>gamma</code> definen la frontera de lo "normal".', hp: ['nu', 'gamma'] },
    { id: 'logistic', name: 'Regresión logística', status: 'soon', paradigm: 'ml', supervision: 'sup', family: 'linear', task: 'Clasificación', blurb: 'Regularización L1/L2 y su efecto en los coeficientes y la calibración.', hp: ['C', 'penalty'] },
    { id: 'autoencoder', name: 'Autoencoder denso', status: 'soon', paradigm: 'dl', supervision: 'self', family: 'autoencoder', task: 'Detección de anomalías', blurb: 'El hermano determinista del VAE: cuello de botella, capacidad y error de reconstrucción.', hp: ['latent_dim', 'hidden', 'epochs'] },
    { id: 'mlp', name: 'Red neuronal (MLP)', status: 'soon', paradigm: 'dl', supervision: 'sup', family: 'convnet', task: 'Clasificación y regresión', blurb: 'Capas, ancho, tasa de aprendizaje, <i>dropout</i> y regularización.', hp: ['hidden_layers', 'learning_rate', 'dropout'] },
    { id: 'lstm', name: 'LSTM para series temporales', status: 'soon', paradigm: 'dl', supervision: 'self', family: 'sequence', task: 'Anomalías en series temporales', blurb: 'Ventana, horizonte y estado oculto para detectar comportamientos anómalos en el tiempo.', hp: ['window', 'hidden_size', 'learning_rate'] },
  ];

  return { PARADIGMS, SUPERVISION, FAMILIES, ALGOS };
});
