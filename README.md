# ML Hyperparameter Lab

Laboratorios interactivos para **entender, de forma gráfica y directa, cómo cada hiperparámetro de un algoritmo de Machine Learning cambia sus resultados**. Pensado como material de clase.

🔗 **Sitio:** https://mirloct.github.io/ml-hyperparameter-lab/

## Algoritmos

| Algoritmo | Estado | Hiperparámetros explorados |
|---|---|---|
| **Isolation Forest** (detección de anomalías) | ✅ Disponible | `n_estimators`, `max_samples`, `contamination`, `max_features`, `bootstrap`, `random_state` |
| K-Means, DBSCAN, Random Forest, SVM, Gradient Boosting | 🕓 Próximamente | — |

## Qué ofrece el laboratorio de Isolation Forest

- **El modelo se entrena en el navegador** (implementación propia en JavaScript, mismas convenciones que `sklearn.ensemble.IsolationForest`). Sin servidor, sin instalación.
- **Cada hiperparámetro tiene su explicación**: qué es, qué controla en el modelo, qué pasa al subirlo/bajarlo, valor típico y dónde se nota. Aparece al señalarlo (tooltip) y al usarlo (panel expandido), con un dato en vivo (p. ej. profundidad máxima `⌈log₂ ψ⌉`, umbral y nº de puntos marcados).
- **Visualizaciones elegidas por lo que enseñan:**
  - *Mapa de scores* 2D con el umbral de decisión (contorno) y los puntos marcados.
  - *Un árbol por dentro*: particiones aleatorias nivel por nivel y el camino de aislamiento de un punto.
  - *Distribución de scores por clase*, *curvas ROC y Precisión–Recall* con el punto operativo actual.
  - *Barrido del hiperparámetro activo* (varias semillas por valor: media, rango y dispersión).
  - *Anatomía de un punto* (h(x) por árbol y la fórmula del score con tus números) y *convergencia* del score con el nº de árboles.
- **Métricas separadas por lo que evalúan:** calidad del *ranking* (ROC-AUC, AP; no dependen del umbral) y de la *decisión* (Precisión, Recall, F1; dependen de `contamination`). Con comparación contra una **referencia fijada** (▲/▼).
- **Escenarios guiados** con datos y parámetros precargados: estabilidad vs. nº de árboles, `contamination` (umbral ≠ ranking), *masking* y `max_samples`, el anillo y `max_features`, `random_state`, y cómo se aísla un punto.
- 5 conjuntos de datos sintéticos, clic en el mapa para **agregar tus propios puntos** (normales o anómalos), tema claro/oscuro, responsive, y la configuración queda en la URL (`#…`) para compartirla.

> Los conjuntos son sintéticos para conocer la etiqueta real y poder evaluar. El modelo nunca ve las etiquetas.

## Ejecutar en local

No hay dependencias ni paso de compilación.

```bash
npm run serve      # http://localhost:8080  (o abre index.html directamente)
npm test           # pruebas del núcleo (Node ≥ 18)
```

## Estructura

```
index.html                         portada / hub de algoritmos
assets/css/                        estilos (tokens claro/oscuro)
assets/js/common.js                tema, tooltips, utilidades
assets/js/core/metrics.js          ROC, PR, AP, matriz de confusión (desde cero)
assets/js/core/datasets.js         generadores de datos sintéticos 2D
algorithms/isolation-forest/
  iforest.js                       el algoritmo (árboles, score, umbral)
  content.js                       definiciones de hiperparámetros, métricas y escenarios
  charts.js                        gráficas SVG a mano
  app.js                           estado y renderizado
tests/                             pruebas del núcleo (node:test)
```

### Agregar un nuevo algoritmo

1. Crea `algorithms/<nombre>/` con su `index.html` (reutiliza `assets/css/site.css` y `assets/js/common.js`).
2. Implementa el modelo y define, como en `content.js`, la definición de cada hiperparámetro.
3. Agrega la tarjeta en `index.html` (cambia "Próximamente" por "Disponible").

## Publicación

El sitio es 100 % estático y se publica con **GitHub Pages** desde la rama `main` (raíz).

## Referencias

- Liu, Ting & Zhou (2008). *Isolation Forest.* IEEE ICDM.
- Liu, Ting & Zhou (2012). *Isolation-based anomaly detection.* ACM TKDD 6(1).
- Hariri, Kind & Brunner (2019). *Extended Isolation Forest.* IEEE TKDE.
- scikit-learn: `sklearn.ensemble.IsolationForest`.

## Licencia

MIT
