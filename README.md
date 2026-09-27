# ML Hyperparameter Lab

Laboratorios interactivos para **entender, de forma gráfica y directa, cómo cada hiperparámetro de un modelo de Machine Learning o Deep Learning cambia sus resultados**. Pensado como material de clase.

🔗 **Sitio:** https://mirloct.github.io/ml-hyperparameter-lab/

## Catálogo

El sitio organiza los modelos en tres ejes: **paradigma** (Machine Learning / Deep Learning), **tipo de supervisión** (supervisado, no supervisado, semi-supervisado, autosupervisado) y **familia metodológica** (aislamiento, bagging, boosting, densidad, centroides, margen, autoencoders, secuencias…). La portada permite filtrar por supervisión.

| Curso | Paradigma · supervisión | Estado |
|---|---|---|
| [**Isolation Forest**](algorithms/isolation-forest/) | ML · no supervisado · aislamiento | ✅ Disponible |
| [**Autoencoder Variacional (VAE)**](algorithms/vae/) | DL · autosupervisado · autoencoders | ✅ Disponible |
| K-Means, DBSCAN, LOF, Random Forest, Gradient Boosting, SVM, One-Class SVM, regresión logística, autoencoder denso, MLP, LSTM | — | 🕓 Próximamente |

Ambos cursos siguen la **misma estructura de 6 capítulos**: la idea → laboratorio → evaluar → optimizar → casos reales → checklist, más bibliografía. Cada capítulo abre con objetivos de aprendizaje y cierra con un resumen y preguntas de comprobación; los términos subrayados muestran su definición al señalarlos.

### Isolation Forest — detección de anomalías
Carrera de aislamiento interactiva, los 6 hiperparámetros con su efecto medido, evaluación **con y sin etiquetas** (Jaccard@k, Mass-Volume, acuerdo con kNN), y una **arena de optimización** que compara grid, random, bayesiana y successive halving con el mismo presupuesto sobre un paisaje tabular precalculado.

### Autoencoder Variacional (VAE) — deep learning en el navegador
Una red que **se entrena de verdad en tu navegador**, con los gradientes derivados a mano (sin librerías): imágenes 8×8 generadas a partir de factores conocidos, reconstrucciones, espacio latente, mapa generativo y KL por dimensión. Explora `latent_dim`, `beta`, capacidad, `learning_rate`, recocido del KL, épocas, lote y semilla. Muestra en vivo el **colapso posterior** y cómo el recocido lo rescata, y una cuadrícula β × latent_dim que demuestra que ambos **interactúan**.

## Estándares compartidos entre modelos

Al añadir un modelo nuevo se reutilizan estas piezas para que el material sea coherente:

- **`assets/js/core/tuning-doc.js`** — el bloque «Qué **no** optimizar», con tres categorías fijas: **fíjalo** (hay un plateau), **decídelo** (es una decisión de negocio) o **solo reproducibilidad** (optimizarlo es sobreajustar al azar). Cada laboratorio declara su `ML.TUNING_DOC` y el bloque se dibuja igual en todos.
- **`assets/js/chapters.js`** — navegación por capítulos, glosario con definición al señalar, resúmenes, preguntas de comprobación y casos reales. Cada laboratorio declara `ML.GLOSSARY`, `ML.CHECKS`, `ML.TAKEAWAYS`, `ML.CASES` y `ML.READING_CONCEPTS`.
- **`assets/css/chapters.css` y `lab.css`** — estructura de capítulos, tarjetas, tablas de indicadores (con los tres estados de «¿necesita etiquetas?») y controles del laboratorio.
- **`assets/js/catalog.js`** — taxonomía y ficha de cada modelo para la portada.

## Ejecutar en local

Sin dependencias de ejecución ni paso de compilación.

```bash
npm run serve          # http://localhost:8080  (o abre index.html directamente)
npm test               # 30 pruebas del núcleo (Node ≥ 18)
```

Herramientas de desarrollo (requieren `puppeteer-core` y un navegador Chromium/Edge; variable `BROWSER` para la ruta):

```bash
npm run e2e              # end-to-end de Isolation Forest
npm run e2e:vae          # end-to-end del VAE (entrena de verdad: ~2 min)
npm run check:reading    # secuencia de lectura de un curso (definición antes de uso, citas, navegación)
npm run build:landscape  # regenera el paisaje de hiperparámetros de Isolation Forest
```

## Estructura

```
index.html                         portada: catálogo por paradigma / familia / supervisión
assets/css/                        site, hub, lab y chapters (compartidos)
assets/js/common.js                tema, tooltips, utilidades
assets/js/catalog.js, hub.js       taxonomía y render de la portada
assets/js/chapters.js              estructura de lectura compartida por los cursos
assets/js/core/metrics.js          ROC, PR, AP, matriz de confusión (desde cero)
assets/js/core/unsup.js            Jaccard, Spearman, Mass-Volume, kNN (indicadores sin etiquetas)
assets/js/core/tuning.js           grid, random, bayesiana (GP-EI), successive halving
assets/js/core/tuning-doc.js       estándar «qué no optimizar»
assets/js/core/datasets.js         generadores de datos sintéticos 2D
algorithms/isolation-forest/       iforest.js · content.js · charts.js · app.js · race.js · tuning-ui.js · landscape.js
algorithms/vae/                    vae.js · shapes.js · content.js · vae-charts.js · app.js · grid.js · vae.css
tools/                             servidor, generador del paisaje, e2e y verificador de lectura
tests/                             pruebas del núcleo (node:test)
```

### Agregar un nuevo modelo

1. Añade su ficha a `assets/js/catalog.js` (paradigma, supervisión, familia, tarea, hiperparámetros).
2. Crea `algorithms/<nombre>/` con su `index.html` siguiendo la estructura de 6 capítulos y enlazando `assets/css/lab.css`, `chapters.css` y `assets/js/chapters.js`.
3. Declara `ML.GLOSSARY`, `ML.CHECKS`, `ML.TAKEAWAYS`, `ML.CASES`, `ML.READING_CONCEPTS` y **`ML.TUNING_DOC`** (usa las mismas tres categorías de «qué no optimizar»).
4. Añade pruebas en `tests/` y comprueba con `npm run check:reading`.

## Publicación

Sitio estático publicado con **GitHub Pages** desde la rama `main` (raíz).

## Referencias principales

**Isolation Forest:** Liu, Ting & Zhou (2008, 2012) · Hariri, Kind & Brunner (2021) · Probst & Boulesteix (2018) · Goix (2016) · Campos et al. (2016) · Marques et al. (2020) · Perini et al. (2023).

**VAE:** Kingma & Welling (2014, 2019) · Rezende et al. (2014) · Higgins et al. (2017, β-VAE) · Bowman et al. (2016, recocido del KL) · Locatello et al. (2019, límites del desenredo) · Lucas et al. (2019, colapso posterior).

**Optimización de hiperparámetros:** Bergstra & Bengio (2012) · Snoek et al. (2012) · Li et al. (2018, Hyperband) · Probst et al. (2019) · Akiba et al. (2019, Optuna) · Bischl et al. (2023).

La lista completa, con indicación de cuáles se consultaron directamente en la fuente, está al final de cada curso.

## Licencia

MIT
