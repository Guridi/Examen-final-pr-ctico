"""
Entrenamiento de la red neuronal sobre el conjunto de datos EMNIST.

Dataset: EMNIST/letters (https://www.tensorflow.org/datasets/catalog/emnist)
    - 88.800 imágenes de entrenamiento y 14.800 de prueba.
    - 26 clases balanceadas (A-Z); mayúsculas y minúsculas de una misma
      letra comparten etiqueta.

Arquitectura (cumple lo pedido: capa de entrada, al menos dos capas ocultas
y una capa de salida). Se ofrecen dos variantes seleccionables por consola:

    --arquitectura cnn   (por defecto) red convolucional
    --arquitectura mlp                perceptrón multicapa totalmente conectado

Uso:
    python src/train_model.py                      # CNN, 20 épocas
    python src/train_model.py --arquitectura mlp   # MLP
    python src/train_model.py --epocas 5           # entrenamiento corto
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from config import (
    ALTO,
    ANCHO,
    CLASES,
    DESPLAZAMIENTO_ETIQUETA,
    DIR_RESULTADOS,
    EPOCAS,
    FORMA_ENTRADA,
    FRACCION_VALIDACION,
    NOMBRE_DATASET,
    NUM_CLASES,
    RUTA_HISTORIAL,
    RUTA_METRICAS,
    RUTA_MODELO,
    SEMILLA,
    TAM_LOTE,
)

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================================================
# 1. Carga y preparación del dataset EMNIST
# ==========================================================================
def _normalizar_tfds(imagen, etiqueta):
    """Corrige la orientación, escala a [0,1] y reindexa la etiqueta.

    Detalle importante de EMNIST: las imágenes vienen almacenadas
    traspuestas (giradas 90° y espejadas) respecto a como se leen. Hay que
    trasponer alto y ancho para que las letras queden derechas y coincidan
    con el formato en que preprocesamos nuestras propias imágenes.

    Las etiquetas de EMNIST/letters van de 1 a 26; las llevamos a 0..25.
    """
    imagen = tf.transpose(imagen, perm=[1, 0, 2])
    imagen = tf.cast(imagen, tf.float32) / 255.0
    etiqueta = tf.cast(etiqueta, tf.int32) - DESPLAZAMIENTO_ETIQUETA
    return imagen, etiqueta


def _normalizar_crudo(imagen, etiqueta):
    """Escala a [0,1] y añade el canal.

    Con la fuente 'raw' la orientación y el reindexado de etiquetas ya los
    resolvió `emnist_data.cargar_split`, así que aquí sólo queda el escalado.
    """
    imagen = tf.cast(imagen, tf.float32) / 255.0
    imagen = tf.expand_dims(imagen, -1)
    return imagen, etiqueta


def _aumentar(imagen, etiqueta):
    """Aumento de datos ligero, sólo sobre el split de entrenamiento.

    Pequeñas rotaciones, traslaciones y zoom hacen que el modelo tolere
    mejor la variabilidad de una foto o de un trazo hecho con el ratón.
    """
    imagen = tf.keras.ops.image.affine_transform(
        imagen[None, ...],
        _matriz_aleatoria(),
        interpolation="bilinear",
        fill_mode="constant",
        fill_value=0.0,
    )[0]
    return imagen, etiqueta


def _matriz_aleatoria():
    """Genera los 8 parámetros de una transformación afín aleatoria suave."""
    angulo = tf.random.uniform([], -12.0, 12.0) * np.pi / 180.0
    escala = tf.random.uniform([], 0.9, 1.1)
    dx = tf.random.uniform([], -2.0, 2.0)
    dy = tf.random.uniform([], -2.0, 2.0)

    cos = tf.cos(angulo) / escala
    sin = tf.sin(angulo) / escala
    cx = (ANCHO - 1) / 2.0
    cy = (ALTO - 1) / 2.0
    # Rotación y escala alrededor del centro, más traslación.
    a0, a1 = cos, -sin
    a2 = cx - cos * cx + sin * cy + dx
    b0, b1 = sin, cos
    b2 = cy - sin * cx - cos * cy + dy
    return tf.stack([[a0, a1, a2, b0, b1, b2, 0.0, 0.0]])


def _empaquetar(ds, entrenamiento: bool, tam_lote: int, aumentar: bool, normalizar):
    ds = ds.map(normalizar, num_parallel_calls=AUTOTUNE)
    if entrenamiento:
        ds = ds.shuffle(10_000, seed=SEMILLA)
        if aumentar:
            ds = ds.map(_aumentar, num_parallel_calls=AUTOTUNE)
    return ds.batch(tam_lote).prefetch(AUTOTUNE)


def _cargar_crudo(tam_lote: int, aumentar: bool):
    """Carga EMNIST desde los ficheros IDX originales (datos completos)."""
    from emnist_data import cargar_split, resumen

    x_train, y_train = cargar_split("train")
    x_test, y_test = cargar_split("test")
    print("  " + resumen("train+val", y_train))
    print("  " + resumen("test     ", y_test))

    # Separamos la validación con una permutación reproducible. El dataset
    # está equilibrado, así que un corte aleatorio mantiene las proporciones.
    generador = np.random.default_rng(SEMILLA)
    orden = generador.permutation(len(x_train))
    n_val = int(len(orden) * FRACCION_VALIDACION)
    idx_val, idx_train = orden[:n_val], orden[n_val:]

    crear = tf.data.Dataset.from_tensor_slices
    ds_train = crear((x_train[idx_train], y_train[idx_train]))
    ds_val = crear((x_train[idx_val], y_train[idx_val]))
    ds_test = crear((x_test, y_test))

    info = {
        "imagenes_entrenamiento": int(len(idx_train)),
        "imagenes_validacion": int(len(idx_val)),
        "imagenes_prueba": int(len(x_test)),
        "clases_en_prueba": int(len(np.unique(y_test))),
        "fuente": "ficheros IDX originales del NIST",
    }
    return (
        _empaquetar(ds_train, True, tam_lote, aumentar, _normalizar_crudo),
        _empaquetar(ds_val, False, tam_lote, aumentar, _normalizar_crudo),
        _empaquetar(ds_test, False, tam_lote, aumentar, _normalizar_crudo),
        info,
    )


def _cargar_tfds(tam_lote: int, aumentar: bool):
    """Carga EMNIST vía TensorFlow Datasets.

    Atención: en TFDS 4.9.x los splits de emnist/letters vienen truncados y
    el de prueba sólo contiene las letras A-S. Se mantiene esta ruta por
    completitud, pero la fuente por defecto es 'raw'.
    """
    pct_val = int(FRACCION_VALIDACION * 100)
    division = [f"train[:{100 - pct_val}%]", f"train[{100 - pct_val}%:]", "test"]
    (bruto_train, bruto_val, bruto_test), info_tfds = tfds.load(
        NOMBRE_DATASET, split=division, as_supervised=True, with_info=True, shuffle_files=True
    )
    print("  [aviso] la fuente TFDS entrega el split de prueba truncado (sólo A-S).")
    info = {
        "imagenes_entrenamiento": int(info_tfds.splits["train"].num_examples),
        "imagenes_prueba": int(info_tfds.splits["test"].num_examples),
        "fuente": "tensorflow_datasets (splits truncados)",
    }
    return (
        _empaquetar(bruto_train, True, tam_lote, aumentar, _normalizar_tfds),
        _empaquetar(bruto_val, False, tam_lote, aumentar, _normalizar_tfds),
        _empaquetar(bruto_test, False, tam_lote, aumentar, _normalizar_tfds),
        info,
    )


def cargar_datasets(tam_lote: int, aumentar: bool = True, fuente: str = "raw"):
    """Devuelve (train, val, test, info) como tf.data.Dataset ya preparados."""
    if fuente == "raw":
        return _cargar_crudo(tam_lote, aumentar)
    return _cargar_tfds(tam_lote, aumentar)


# ==========================================================================
# 2. Definición de la red neuronal
# ==========================================================================
def construir_cnn() -> tf.keras.Model:
    """Red convolucional: entrada + 4 bloques ocultos + salida softmax."""
    capas = tf.keras.layers
    return tf.keras.Sequential(
        [
            # ---- CAPA DE ENTRADA -----------------------------------------
            capas.Input(shape=FORMA_ENTRADA, name="entrada"),
            # ---- CAPA OCULTA 1: extracción de bordes ---------------------
            capas.Conv2D(32, 3, padding="same", activation="relu", name="oculta1_conv"),
            capas.BatchNormalization(),
            capas.Conv2D(32, 3, padding="same", activation="relu", name="oculta1b_conv"),
            capas.BatchNormalization(),
            capas.MaxPooling2D(2),
            capas.Dropout(0.25),
            # ---- CAPA OCULTA 2: formas compuestas ------------------------
            capas.Conv2D(64, 3, padding="same", activation="relu", name="oculta2_conv"),
            capas.BatchNormalization(),
            capas.Conv2D(64, 3, padding="same", activation="relu", name="oculta2b_conv"),
            capas.BatchNormalization(),
            capas.MaxPooling2D(2),
            capas.Dropout(0.25),
            # ---- CAPA OCULTA 3: densa totalmente conectada ---------------
            capas.Flatten(),
            capas.Dense(256, activation="relu", name="oculta3_densa"),
            capas.BatchNormalization(),
            capas.Dropout(0.5),
            # ---- CAPA DE SALIDA ------------------------------------------
            capas.Dense(NUM_CLASES, activation="softmax", name="salida"),
        ],
        name="cnn_emnist",
    )


def construir_mlp() -> tf.keras.Model:
    """Perceptrón multicapa: entrada + 3 capas ocultas densas + salida."""
    capas = tf.keras.layers
    return tf.keras.Sequential(
        [
            # ---- CAPA DE ENTRADA -----------------------------------------
            capas.Input(shape=FORMA_ENTRADA, name="entrada"),
            capas.Flatten(name="aplanado"),
            # ---- CAPA OCULTA 1 -------------------------------------------
            capas.Dense(512, activation="relu", name="oculta1"),
            capas.BatchNormalization(),
            capas.Dropout(0.3),
            # ---- CAPA OCULTA 2 -------------------------------------------
            capas.Dense(256, activation="relu", name="oculta2"),
            capas.BatchNormalization(),
            capas.Dropout(0.3),
            # ---- CAPA OCULTA 3 -------------------------------------------
            capas.Dense(128, activation="relu", name="oculta3"),
            capas.Dropout(0.3),
            # ---- CAPA DE SALIDA ------------------------------------------
            capas.Dense(NUM_CLASES, activation="softmax", name="salida"),
        ],
        name="mlp_emnist",
    )


def construir_modelo(arquitectura: str) -> tf.keras.Model:
    modelo = construir_cnn() if arquitectura == "cnn" else construir_mlp()
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


# ==========================================================================
# 3. Gráficos de evaluación
# ==========================================================================
def graficar_historial(historial: dict, ruta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(historial["accuracy"], label="entrenamiento")
    ax1.plot(historial["val_accuracy"], label="validación")
    ax1.set_title("Exactitud por época")
    ax1.set_xlabel("época")
    ax1.set_ylabel("exactitud")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(historial["loss"], label="entrenamiento")
    ax2.plot(historial["val_loss"], label="validación")
    ax2.set_title("Pérdida por época")
    ax2.set_xlabel("época")
    ax2.set_ylabel("pérdida")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def graficar_matriz_confusion(y_real, y_pred, ruta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matriz = tf.math.confusion_matrix(y_real, y_pred, num_classes=NUM_CLASES).numpy()
    normalizada = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(normalizada, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASES), CLASES, fontsize=8)
    ax.set_yticks(range(NUM_CLASES), CLASES, fontsize=8)
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión normalizada — conjunto de prueba EMNIST")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return matriz


def graficar_muestras_test(modelo, ds_test, ruta, n=24):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs, etiquetas = next(iter(ds_test.unbatch().batch(n)))
    probas = modelo.predict(imgs, verbose=0)
    pred = probas.argmax(axis=1)

    filas, cols = 3, 8
    fig, ejes = plt.subplots(filas, cols, figsize=(cols * 1.5, filas * 1.9))
    for i, ax in enumerate(ejes.flat):
        ax.imshow(imgs[i].numpy().squeeze(), cmap="gray")
        ax.axis("off")
        ok = pred[i] == etiquetas[i].numpy()
        ax.set_title(
            f"real {CLASES[etiquetas[i]]} / pred {CLASES[pred[i]]}\n{probas[i].max():.1%}",
            fontsize=7,
            color="green" if ok else "red",
        )
    fig.suptitle("Predicciones sobre el conjunto de prueba EMNIST", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


# ==========================================================================
# 4. Programa principal
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Entrena la red neuronal con EMNIST")
    parser.add_argument("--arquitectura", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--epocas", type=int, default=EPOCAS)
    parser.add_argument("--lote", type=int, default=TAM_LOTE)
    parser.add_argument("--sin-aumento", action="store_true", help="desactiva el aumento de datos")
    parser.add_argument("--salida", default=None, help="ruta alternativa para guardar el modelo")
    parser.add_argument(
        "--fuente",
        choices=["raw", "tfds"],
        default="raw",
        help="origen de los datos: 'raw' lee los ficheros IDX completos del NIST "
        "(recomendado); 'tfds' usa TensorFlow Datasets, cuyos splits vienen truncados",
    )
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEMILLA)
    ruta_modelo = args.salida or RUTA_MODELO

    print("=" * 70)
    print(f"Cargando {NOMBRE_DATASET} (fuente: {args.fuente}) ...")
    ds_train, ds_val, ds_test, info = cargar_datasets(
        args.lote, aumentar=not args.sin_aumento, fuente=args.fuente
    )
    n_train = info["imagenes_entrenamiento"]
    n_test = info["imagenes_prueba"]
    print(f"  entrenamiento: {n_train:,} imágenes")
    print(f"  prueba:        {n_test:,} imágenes")
    print(f"  clases:        {NUM_CLASES} ({CLASES[0]}..{CLASES[-1]})")

    print("=" * 70)
    modelo = construir_modelo(args.arquitectura)
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]

    print("=" * 70)
    print(f"Entrenando ({args.arquitectura.upper()}, {args.epocas} épocas máx.) ...")
    inicio = time.time()
    historia = modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epocas,
        callbacks=callbacks,
        verbose=2,
    )
    duracion = time.time() - inicio
    print(f"Entrenamiento terminado en {duracion / 60:.1f} minutos.")

    # ---- Evaluación sobre el conjunto de PRUEBA de EMNIST ----------------
    print("=" * 70)
    print("Evaluando sobre el conjunto de prueba de EMNIST ...")
    perdida_test, exactitud_test = modelo.evaluate(ds_test, verbose=0)
    print(f"  pérdida  de prueba: {perdida_test:.4f}")
    print(f"  exactitud de prueba: {exactitud_test:.4f}  ({exactitud_test:.2%})")

    # Predicciones completas para métricas detalladas
    y_real = np.concatenate([y.numpy() for _, y in ds_test])
    probas = modelo.predict(ds_test, verbose=0)
    y_pred = probas.argmax(axis=1)

    # Exactitud top-3: útil para saber si el acierto está "cerca"
    top3 = tf.math.in_top_k(y_real, probas, k=3).numpy().mean()
    print(f"  exactitud top-3:     {top3:.4f}  ({top3:.2%})")

    # ---- Guardado --------------------------------------------------------
    modelo.save(ruta_modelo)
    print(f"Modelo guardado en {ruta_modelo}")

    sufijo = f"_{args.arquitectura}"
    graficar_historial(historia.history, DIR_RESULTADOS / f"historial_entrenamiento{sufijo}.png")
    matriz = graficar_matriz_confusion(
        y_real, y_pred, DIR_RESULTADOS / f"matriz_confusion{sufijo}.png"
    )
    graficar_muestras_test(modelo, ds_test, DIR_RESULTADOS / f"muestras_test{sufijo}.png")

    # Sólo tienen sentido las clases con ejemplos en el conjunto de prueba:
    # una clase ausente daría 0 % y ensuciaría el ranking de las peores.
    soporte = matriz.sum(axis=1)
    exactitud_por_clase = {
        CLASES[i]: float(matriz[i, i] / soporte[i]) for i in range(NUM_CLASES) if soporte[i] > 0
    }
    ausentes = [CLASES[i] for i in range(NUM_CLASES) if soporte[i] == 0]
    if ausentes:
        print(f"  [aviso] clases sin ejemplos en el test: {', '.join(ausentes)}")
    peores = sorted(exactitud_por_clase.items(), key=lambda kv: kv[1])[:5]
    print("  clases más difíciles: " + ", ".join(f"{c} ({v:.1%})" for c, v in peores))

    metricas = {
        "arquitectura": args.arquitectura,
        "dataset": NOMBRE_DATASET,
        "fuente_datos": info["fuente"],
        "clases_en_prueba": len(exactitud_por_clase),
        "num_clases": NUM_CLASES,
        "parametros": int(modelo.count_params()),
        "epocas_ejecutadas": len(historia.history["loss"]),
        "duracion_minutos": round(duracion / 60, 2),
        "perdida_test": float(perdida_test),
        "exactitud_test": float(exactitud_test),
        "exactitud_top3_test": float(top3),
        "exactitud_val_final": float(historia.history["val_accuracy"][-1]),
        "imagenes_entrenamiento": int(n_train),
        "imagenes_prueba": int(n_test),
        "exactitud_por_clase": exactitud_por_clase,
    }
    RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    RUTA_HISTORIAL.write_text(
        json.dumps({k: [float(x) for x in v] for k, v in historia.history.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Métricas guardadas en {RUTA_METRICAS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
