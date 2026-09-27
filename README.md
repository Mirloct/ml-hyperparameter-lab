# ML Hyperparameter Lab

Laboratorios interactivos para **entender, de forma gráfica y directa, cómo cada hiperparámetro de un algoritmo de Machine Learning cambia sus resultados**. Pensado como material de clase.

🔗 **Sitio:** https://mirloct.github.io/ml-hyperparameter-lab/

## Algoritmos

| Algoritmo | Estado | Contenido |
|---|---|---|
| **Isolation Forest** (detección de anomalías) | ✅ Disponible | Curso de 6 capítulos + bibliografía |
| K-Means, DBSCAN, Random Forest, SVM, Gradient Boosting | 🕓 Próximamente | — |

## Isolation Forest: ruta de lectura

1. **La idea** — *Carrera de aislamiento* interactiva: cuántos cortes aleatorios hacen falta para dejar solo a un punto anómalo frente a uno normal, y cómo eso se convierte en el score `s(x) = 2^(−E[h]/c(ψ))`.
2. **Laboratorio** — el modelo se entrena en tu navegador. Mueve `n_estimators`, `max_samples`, `contamination`, `max_features`, `bootstrap` y `random_state`; cada uno tiene definición, rol en el modelo, efecto de subirlo/bajarlo, valor típico, **caso real** y **prioridad de tuneo**. Mapa de scores, vista de un árbol, histograma de scores, ROC/PR, **barrido** del hiperparámetro activo (con costo en ms y detección del *plateau*), anatomía de un punto y convergencia. Incluye 7 escenarios guiados.
3. **Evaluar** — indicadores **con etiquetas** (ROC-AUC, AP, Precisión@k, F1) y **sin etiquetas** (estabilidad **Jaccard@k** entre semillas, correlación de rangos, **Mass-Volume**, acuerdo con otro detector). Tabla "qué indicador usar y cuándo" y una comprobación empírica de **si esos indicadores realmente sirven** para elegir configuraciones (spoiler: tienen puntos ciegos).
4. **Optimizar** — qué es un hiperparámetro y por qué tunear, cuánto se gana en cada dataset (*tunabilidad*), qué **no** optimizar (`n_estimators`, semilla, `contamination`), métodos (manual, grid, random, bayesiana, *successive halving*/Hyperband, evolutivos) y una **arena** que compara los métodos con el mismo presupuesto, con objetivo supervisado o sin etiquetas.
5. **Casos reales** — fraude con tarjetas, mantenimiento predictivo, ciberseguridad, calidad industrial, monitoreo clínico y limpieza de datos, cada uno con su decisión de negocio, indicador y trampa típica. Botón para simularlo en el laboratorio.
6. **Checklist** — flujo de trabajo, errores comunes y límites.
7. **Bibliografía** — 22 referencias (las marcadas con ✔ se consultaron en la fuente al prepararlas).

Cada capítulo empieza con objetivos de aprendizaje y termina con un resumen y preguntas de comprobación. Los términos subrayados muestran su definición al señalarlos.

> Los conjuntos son sintéticos para conocer la etiqueta real y poder evaluar. El modelo nunca ve las etiquetas.

## Ejecutar en local

No hay dependencias de ejecución ni paso de compilación.

```bash
npm run serve          # http://localhost:8080  (o abre index.html directamente)
npm test               # 18 pruebas del núcleo (Node ≥ 18)
```

Herramientas de desarrollo (requieren `puppeteer-core` y un navegador Chromium/Edge; variable `BROWSER` para la ruta):

```bash
npm run e2e            # prueba end-to-end de controles, escenarios, arena y casos
npm run check:reading  # valida la secuencia de lectura (definición antes de uso, citas, navegación)
npm run build:landscape  # regenera el paisaje de hiperparámetros (≈ 1 min)
```

## Estructura

```
index.html                         portada / hub de algoritmos
assets/css/                        estilos (tokens claro/oscuro)
assets/js/common.js                tema, tooltips, utilidades
assets/js/core/metrics.js          ROC, PR, AP, matriz de confusión (desde cero)
assets/js/core/unsup.js            Jaccard, Spearman, Mass-Volume/Excess-Mass, kNN (indicadores sin etiquetas)
assets/js/core/tuning.js           grid, random, bayesiana (GP-EI) y successive halving sobre un paisaje tabular
assets/js/core/datasets.js         generadores de datos sintéticos 2D
algorithms/isolation-forest/
  index.html                       el curso (capítulos)
  iforest.js                       el algoritmo (árboles, score, umbral)
  content.js                       hiperparámetros, métricas, escenarios, casos, preguntas y glosario
  charts.js, charts2.js            gráficas SVG a mano (redibujo automático al cambiar el ancho)
  app.js                           laboratorio: estado y renderizado
  race.js                          carrera de aislamiento (capítulo 1)
  tuning-ui.js                     arena de optimización e indicadores sin etiquetas (capítulos 3 y 4)
  chapters.js                      navegación, glosario, resúmenes y casos
  landscape.js                     paisaje precalculado (generado por tools/build-landscape.js)
tools/                             servidor, generador del paisaje, e2e y verificador de lectura
tests/                             pruebas del núcleo (node:test)
```

### Agregar un nuevo algoritmo

1. Crea `algorithms/<nombre>/` con su `index.html` (reutiliza `assets/css/site.css` y `assets/js/common.js`).
2. Implementa el modelo y define, como en `content.js`, la definición de cada hiperparámetro.
3. Agrega la tarjeta en `index.html` (cambia "Próximamente" por "Disponible") y pasa `tools/reading-check.js`.

## Publicación

El sitio es 100 % estático y se publica con **GitHub Pages** desde la rama `main` (raíz).

## Referencias principales

Liu, Ting & Zhou (2008, 2012) · Hariri, Kind & Brunner (2021, *IEEE TKDE* 33(4)) · Probst & Boulesteix (2018) · Probst, Boulesteix & Bischl (2019) · Bergstra & Bengio (2012) · Snoek et al. (2012) · Li et al. (2018) · Akiba et al. (2019) · Goix (2016) · Campos et al. (2016) · Marques et al. (2020) · Domingues et al. (2018) · Perini et al. (2023) · Saito & Rehmsmeier (2015). La lista completa está al final del curso.

## Licencia

MIT
