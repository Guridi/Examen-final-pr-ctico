# Reconocimiento de letras manuscritas con EMNIST

Examen final práctico. Red neuronal entrenada con el conjunto de datos
[EMNIST](https://www.tensorflow.org/datasets/catalog/emnist) que reconoce letras
manuscritas, más una página web con un recuadro donde escribir una letra y un
servidor en Python que responde con la predicción.

---

## Contenido

| Ruta | Descripción |
|---|---|
| `src/config.py` | Configuración compartida: rutas, clases, tamaños, hiperparámetros. |
| `src/train_model.py` | Carga EMNIST, construye la red, la entrena y la evalúa sobre el conjunto de prueba. |
| `src/preprocess.py` | Preprocesamiento de imágenes propias al formato EMNIST. |
| `src/generate_samples.py` | Genera imágenes propias de ejemplo (una por letra). |
| `src/predict_images.py` | Predice las imágenes propias y genera las figuras de resultados. |
| `src/server.py` | Servidor Flask que recibe la imagen y devuelve la predicción. |
| `src/generate_report.py` | Construye el informe en Word con las predicciones. |
| `web/index.html` | Página con el recuadro de escritura y el resultado debajo. |
| `imagenes_propias/` | Imágenes propias a predecir. |
| `modelo/` | Modelo entrenado y métricas. |
| `resultados/` | Figuras, JSON de predicciones e informe Word. |

---

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate          # en fish:  source .venv/bin/activate.fish
pip install -r requirements.txt
```

---

## Uso

### 1. Entrenar el modelo con EMNIST

```bash
python src/train_model.py
```

La primera ejecución descarga EMNIST (~536 MB) a `~/tensorflow_datasets/`.
Opciones útiles:

```bash
python src/train_model.py --arquitectura mlp   # perceptrón multicapa
python src/train_model.py --epocas 5           # entrenamiento corto
python src/train_model.py --sin-aumento        # sin aumento de datos
```

Genera `modelo/modelo_emnist.keras`, `modelo/metricas.json` y las gráficas de
entrenamiento, matriz de confusión y muestras de prueba en `resultados/`.

### 2. Preparar las imágenes propias

Coloca tus imágenes en `imagenes_propias/`. **El nombre del archivo debe empezar
por la letra real**, por ejemplo `A_01.jpg`, `b-2.png`, `Z.png`. Esa letra se usa
como etiqueta real para medir el acierto.

Si quieres un juego de imágenes de ejemplo listo para probar:

```bash
python src/generate_samples.py --limpiar
```

### 3. Predecir las imágenes propias

```bash
python src/predict_images.py
```

Muestra por consola y guarda en `resultados/` la imagen procesada, la etiqueta
real, la etiqueta predicha y la probabilidad de cada imagen.

### 4. Levantar la página web y el servidor

```bash
python src/server.py
```

Abre <http://127.0.0.1:5000>, escribe una letra en el recuadro y el resultado
aparece justo debajo.

### 5. Generar el informe en Word

```bash
python src/generate_report.py
```

Produce `resultados/Informe_Reconocimiento_Letras_EMNIST.docx`.

---

## Detalles de implementación

### El dataset

Se usa la variante **EMNIST/letters**: 88.800 imágenes de entrenamiento y 14.800
de prueba, repartidas en 26 clases balanceadas (A–Z). Mayúsculas y minúsculas de
una misma letra comparten etiqueta.

Dos detalles del dataset que hay que tratar en el código:

- **Las imágenes vienen traspuestas.** EMNIST se distribuye con las imágenes
  giradas 90° y espejadas. En `_normalizar()` se aplica una trasposición para
  dejarlas derechas.
- **Las etiquetas van de 1 a 26**, no de 0 a 25. Se les resta 1 para que
  encajen con la capa softmax.

### La arquitectura

Cumple lo pedido —capa de entrada, al menos dos capas ocultas y capa de salida—
y está disponible en dos variantes:

- `--arquitectura cnn` (por defecto): entrada 28×28×1, dos bloques
  convolucionales (32 y 64 filtros) con BatchNorm/MaxPooling/Dropout, una capa
  densa oculta de 256 unidades y salida softmax de 26.
- `--arquitectura mlp`: entrada aplanada, tres capas densas ocultas
  (512 → 256 → 128) y salida softmax de 26.

### El preprocesamiento

`src/preprocess.py` aplica los cuatro pasos exigidos:

1. **Escala de grises.**
2. **Redimensionamiento a 28×28.**
3. **Normalización** al rango [0, 1].
4. **Inversión de colores condicional**: se mide el brillo del borde de la
   imagen; si el fondo es claro, se invierte para dejar trazo blanco sobre fondo
   negro, que es el convenio de EMNIST.

Además replica la normalización geométrica del propio EMNIST: recorta el
carácter, lo escala a una caja de 20×20 conservando la proporción y lo centra por
su centro de masa dentro del lienzo de 28×28. Este paso es el que más influye en
que una foto real se prediga bien.

El mismo módulo lo usan tanto `predict_images.py` como `server.py`, así que el
formato de entrada es idéntico en los dos caminos.

---

## Notas

- Las imágenes de `imagenes_propias/` generadas por `generate_samples.py` son
  caracteres renderizados con distintas tipografías y deformados para simular una
  fotografía (fondo de papel, iluminación irregular, rotación, desenfoque). Son
  material de relleno reproducible: puedes sustituirlas por fotos de tu propia
  escritura manteniendo la convención de nombres.
- El error residual sobre EMNIST se concentra en pares de letras genuinamente
  ambiguos cuando se escriben aisladas: I/L, G/Q, U/V.
