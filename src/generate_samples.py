"""
Generador de imágenes propias de prueba.

Crea un juego de imágenes (una por cada letra A-Z) que NO provienen de EMNIST
y que simulan fotografías/escaneos reales: fondo de papel con iluminación
irregular, tinta oscura, ligera rotación, desenfoque y tamaños distintos de
28x28. Sirven para ejercitar de punta a punta el preprocesamiento:

    - tienen color y hay que pasarlas a escala de grises,
    - no miden 28x28 y hay que redimensionarlas,
    - la mayoría son tinta oscura sobre fondo claro, así que hay que
      invertirlas para que coincidan con el formato de EMNIST,
    - unas pocas ya vienen en blanco sobre negro, para comprobar que la
      inversión sólo se aplica cuando hace falta.

IMPORTANTE: son imágenes de relleno para que el sistema sea reproducible.
Puedes borrarlas y colocar en `imagenes_propias/` tus propias fotos escritas
a mano; basta con que el nombre del archivo empiece por la letra real
(por ejemplo `A_01.jpg`, `b-2.png`).

Uso:
    python src/generate_samples.py            # 26 imágenes, una por letra
    python src/generate_samples.py --por-letra 2
"""

from __future__ import annotations

import argparse
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import CLASES, DIR_IMAGENES_PROPIAS, SEMILLA

FUENTES_CANDIDATAS = [
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/noto/NotoSans-Medium.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
]


def fuentes_disponibles() -> list[str]:
    from pathlib import Path

    disponibles = [f for f in FUENTES_CANDIDATAS if Path(f).exists()]
    if not disponibles:
        raise RuntimeError("No se encontró ninguna fuente TTF utilizable en el sistema.")
    return disponibles


def _fondo_de_papel(lado: int, rng: random.Random) -> Image.Image:
    """Fondo claro con iluminación irregular y grano, como una foto real."""
    base = rng.randint(225, 250)
    y, x = np.mgrid[0:lado, 0:lado].astype(np.float32)
    # Gradiente diagonal suave = iluminación desigual
    gradiente = (x / lado) * rng.uniform(-22, 22) + (y / lado) * rng.uniform(-22, 22)
    ruido = np.random.normal(0, 3.5, (lado, lado))
    papel = np.clip(base + gradiente + ruido, 0, 255).astype(np.uint8)
    return Image.fromarray(papel, mode="L").convert("RGB")


def crear_imagen_letra(letra: str, rng: random.Random, fuentes: list[str]) -> Image.Image:
    lado = rng.randint(150, 320)
    papel = _fondo_de_papel(lado, rng)

    # Glifo: mayúscula casi siempre, minúscula de vez en cuando (EMNIST/letters
    # une ambas cajas bajo la misma etiqueta).
    glifo = letra if rng.random() < 0.75 else letra.lower()
    ruta_fuente = rng.choice(fuentes)
    tam = int(lado * rng.uniform(0.55, 0.72))
    fuente = ImageFont.truetype(ruta_fuente, tam)

    # Dibujamos la tinta en una capa transparente para poder rotarla sin
    # arrastrar el fondo.
    capa = Image.new("RGBA", (lado, lado), (0, 0, 0, 0))
    dibujo = ImageDraw.Draw(capa)
    tono = rng.randint(15, 65)
    grosor = rng.choice([0, 0, 1, 2])
    caja = dibujo.textbbox((0, 0), glifo, font=fuente, stroke_width=grosor)
    cx = (lado - (caja[2] - caja[0])) / 2 - caja[0]
    cy = (lado - (caja[3] - caja[1])) / 2 - caja[1]
    dibujo.text(
        (cx, cy),
        glifo,
        font=fuente,
        fill=(tono, tono, tono, 255),
        stroke_width=grosor,
        stroke_fill=(tono, tono, tono, 255),
    )

    capa = capa.rotate(rng.uniform(-9, 9), resample=Image.BICUBIC, expand=False)
    imagen = Image.alpha_composite(papel.convert("RGBA"), capa).convert("RGB")
    imagen = imagen.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))

    # Un 20% de las imágenes se entregan ya en blanco sobre negro para
    # comprobar que la inversión es condicional y no incondicional.
    if rng.random() < 0.2:
        imagen = Image.eval(imagen, lambda v: 255 - v)

    # Tamaño final no cuadrado a veces, para forzar el redimensionado.
    if rng.random() < 0.3:
        imagen = imagen.resize((lado, int(lado * rng.uniform(0.8, 1.25))), Image.LANCZOS)
    return imagen


def main():
    parser = argparse.ArgumentParser(description="Genera imágenes propias de prueba")
    parser.add_argument("--por-letra", type=int, default=1, help="imágenes por cada letra")
    parser.add_argument("--limpiar", action="store_true", help="borra las imágenes previas")
    args = parser.parse_args()

    rng = random.Random(SEMILLA)
    np.random.seed(SEMILLA)
    fuentes = fuentes_disponibles()

    if args.limpiar:
        for viejo in DIR_IMAGENES_PROPIAS.glob("*.png"):
            viejo.unlink()

    total = 0
    for letra in CLASES:
        for k in range(args.por_letra):
            imagen = crear_imagen_letra(letra, rng, fuentes)
            destino = DIR_IMAGENES_PROPIAS / f"{letra}_{k + 1:02d}.png"
            imagen.save(destino)
            total += 1

    print(f"Generadas {total} imágenes en {DIR_IMAGENES_PROPIAS}")
    print("Puedes sustituirlas por fotos propias: el nombre debe empezar por la letra real.")


if __name__ == "__main__":
    main()
