"""
Predicción sobre las imágenes propias usando el modelo entrenado con EMNIST.

Para cada imagen se muestra y se guarda:
    - la imagen procesada (28x28, escala de grises, normalizada, invertida
      si hacía falta),
    - la etiqueta real (deducida del nombre del archivo),
    - la etiqueta predicha por el modelo,
    - la probabilidad asociada a esa predicción.

Salidas generadas en `resultados/`:
    predicciones_propias.png    resumen en cuadrícula de todas las imágenes
    detalle/<archivo>.png       ficha por imagen (original, procesada, top-5)
    predicciones.json           datos crudos, usados luego por el informe Word

Uso:
    python src/predict_images.py
    python src/predict_images.py --sin-centrado   # preprocesado simple
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    CLASES,
    DIR_IMAGENES_PROPIAS,
    DIR_RESULTADOS,
    RUTA_MODELO,
    indice_a_letra,
)
from preprocess import cargar_imagenes_propias

DIR_DETALLE = DIR_RESULTADOS / "detalle"
RUTA_PREDICCIONES = DIR_RESULTADOS / "predicciones.json"


def cargar_modelo(ruta=RUTA_MODELO):
    if not ruta.exists():
        raise SystemExit(
            f"No se encontró el modelo en {ruta}.\n"
            "Entrena primero la red con:  python src/train_model.py"
        )
    return tf.keras.models.load_model(ruta)


def predecir_muestras(modelo, muestras: list[dict]) -> list[dict]:
    """Ejecuta el modelo sobre todas las muestras ya preprocesadas."""
    lote = np.stack([m["arreglo"] for m in muestras])[..., None].astype(np.float32)
    probabilidades = modelo.predict(lote, verbose=0)

    resultados = []
    for muestra, probas in zip(muestras, probabilidades):
        indice = int(np.argmax(probas))
        orden = np.argsort(probas)[::-1][:5]
        resultados.append(
            {
                **muestra,
                "etiqueta_predicha": indice_a_letra(indice),
                "probabilidad": float(probas[indice]),
                "top5": [(indice_a_letra(i), float(probas[i])) for i in orden],
                "probas": probas,
            }
        )
    return resultados


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
def figura_resumen(resultados: list[dict], ruta):
    """Cuadrícula con la imagen procesada, etiqueta real, predicha y probabilidad."""
    n = len(resultados)
    cols = 7
    filas = (n + cols - 1) // cols
    fig, ejes = plt.subplots(filas, cols, figsize=(cols * 2.0, filas * 2.5))
    ejes = np.atleast_1d(ejes).ravel()

    for ax in ejes[n:]:
        ax.axis("off")

    for ax, r in zip(ejes, resultados):
        ax.imshow(r["arreglo"], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        real = r["etiqueta_real"] or "?"
        acierto = r["etiqueta_real"] is not None and r["etiqueta_predicha"] == r["etiqueta_real"]
        color = "green" if acierto else ("red" if r["etiqueta_real"] else "gray")
        for lado in ax.spines.values():
            lado.set_edgecolor(color)
            lado.set_linewidth(2.0)
        ax.set_title(
            f"real: {real}   pred: {r['etiqueta_predicha']}\n{r['probabilidad']:.1%}",
            fontsize=9,
            color=color,
        )

    fig.suptitle("Predicciones sobre las imágenes propias", fontsize=13)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close(fig)


def figura_detalle(resultado: dict, ruta):
    """Ficha individual: original, procesada y las 5 clases más probables."""
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3.2))

    ax0.imshow(resultado["pasos"]["original"])
    ax0.set_title("Imagen original", fontsize=10)
    ax0.axis("off")

    ax1.imshow(resultado["arreglo"], cmap="gray", vmin=0, vmax=1)
    invertida = "sí" if resultado["pasos"]["se_invirtio"] else "no"
    ax1.set_title(f"Procesada 28x28\n(invertida: {invertida})", fontsize=10)
    ax1.axis("off")

    letras = [t[0] for t in resultado["top5"]][::-1]
    valores = [t[1] for t in resultado["top5"]][::-1]
    colores = ["#2a9d8f"] * len(valores)
    colores[-1] = "#e76f51"
    ax2.barh(letras, valores, color=colores)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("probabilidad")
    ax2.set_title("5 clases más probables", fontsize=10)
    for i, v in enumerate(valores):
        ax2.text(min(v + 0.02, 0.86), i, f"{v:.1%}", va="center", fontsize=8)

    real = resultado["etiqueta_real"] or "desconocida"
    acierto = (
        resultado["etiqueta_real"] is not None
        and resultado["etiqueta_predicha"] == resultado["etiqueta_real"]
    )
    marca = "correcto" if acierto else "incorrecto"
    color = "green" if acierto else ("red" if resultado["etiqueta_real"] else "black")
    fig.suptitle(
        f"{resultado['nombre']}  —  real: {real}   |   "
        f"predicha: {resultado['etiqueta_predicha']}   |   "
        f"probabilidad: {resultado['probabilidad']:.2%}   ({marca})",
        fontsize=11,
        color=color,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predice las imágenes propias")
    parser.add_argument("--directorio", default=str(DIR_IMAGENES_PROPIAS))
    parser.add_argument("--modelo", default=str(RUTA_MODELO))
    parser.add_argument(
        "--sin-centrado",
        action="store_true",
        help="usa el preprocesado simple (sin centrado por centro de masa)",
    )
    args = parser.parse_args()

    from pathlib import Path

    modelo = cargar_modelo(Path(args.modelo))
    muestras = cargar_imagenes_propias(Path(args.directorio), centrar=not args.sin_centrado)
    if not muestras:
        raise SystemExit(
            f"No hay imágenes en {args.directorio}.\n"
            "Genera las de ejemplo con:  python src/generate_samples.py"
        )

    resultados = predecir_muestras(modelo, muestras)

    DIR_DETALLE.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"{'archivo':<16}{'real':>6}{'predicha':>10}{'probabilidad':>14}   resultado")
    print("-" * 78)

    aciertos = 0
    con_etiqueta = 0
    for r in resultados:
        real = r["etiqueta_real"]
        if real is not None:
            con_etiqueta += 1
            ok = r["etiqueta_predicha"] == real
            aciertos += int(ok)
            estado = "OK" if ok else "FALLO"
        else:
            estado = "-"
        print(
            f"{r['nombre']:<16}{real or '?':>6}{r['etiqueta_predicha']:>10}"
            f"{r['probabilidad']:>13.2%}   {estado}"
        )
        figura_detalle(r, DIR_DETALLE / f"{r['ruta'].stem}.png")

    print("-" * 78)
    if con_etiqueta:
        exactitud = aciertos / con_etiqueta
        print(f"Aciertos: {aciertos}/{con_etiqueta}  ({exactitud:.2%})")
    else:
        exactitud = None
        print("Ninguna imagen tenía etiqueta real deducible del nombre.")

    ruta_resumen = DIR_RESULTADOS / "predicciones_propias.png"
    figura_resumen(resultados, ruta_resumen)
    print(f"Resumen gráfico:  {ruta_resumen}")
    print(f"Fichas por imagen: {DIR_DETALLE}")

    # Datos serializados para construir el informe de Word
    RUTA_PREDICCIONES.write_text(
        json.dumps(
            {
                "exactitud_imagenes_propias": exactitud,
                "aciertos": aciertos,
                "total_con_etiqueta": con_etiqueta,
                "total_imagenes": len(resultados),
                "predicciones": [
                    {
                        "archivo": r["nombre"],
                        "ruta": str(r["ruta"]),
                        "detalle": str(DIR_DETALLE / f"{r['ruta'].stem}.png"),
                        "etiqueta_real": r["etiqueta_real"],
                        "etiqueta_predicha": r["etiqueta_predicha"],
                        "probabilidad": r["probabilidad"],
                        "se_invirtio": bool(r["pasos"]["se_invirtio"]),
                        "top5": r["top5"],
                    }
                    for r in resultados
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Datos JSON:        {RUTA_PREDICCIONES}")
    print("=" * 78)


if __name__ == "__main__":
    main()
