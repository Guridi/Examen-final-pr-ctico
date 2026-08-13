"""
Carga de EMNIST/letters a partir de los archivos originales del NIST.

MOTIVO DE ESTE MÓDULO
---------------------
La versión de EMNIST/letters que sirve TensorFlow Datasets (4.9.10) declara
unos tamaños de split incorrectos y entrega los datos truncados:

                        real (NIST)        entregado por TFDS
    entrenamiento       124.800            88.800
    prueba               20.800            14.800

En entrenamiento el truncamiento pasa desapercibido porque el archivo de
origen viene barajado y las 26 clases siguen equilibradas. Pero el archivo
de prueba está **ordenado por clase**, así que quedarse con los primeros
14.800 ejemplos elimina por completo las siete últimas letras: el split de
prueba de TFDS sólo contiene de la A a la S.

Evaluar sobre ese split da una exactitud que ignora 7 de las 26 clases.
Por eso aquí se leen directamente los ficheros IDX originales, que sí están
completos y equilibrados (4.800 por clase en train, 800 por clase en test).

Los ficheros los descarga el propio TFDS al preparar el dataset, así que se
reutilizan si ya están en disco; si no, se lanza la descarga.

Formato IDX (el mismo de MNIST):
    idx1 (etiquetas): magic(4) | n(4) | n bytes
    idx3 (imágenes):  magic(4) | n(4) | alto(4) | ancho(4) | n*alto*ancho bytes
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from config import ALTO, ANCHO, DESPLAZAMIENTO_ETIQUETA, NUM_CLASES

RAIZ_TFDS = Path.home() / "tensorflow_datasets"
PATRON = "emnist-letters-{split}-{tipo}-idx{n}-ubyte.gz"


# --------------------------------------------------------------------------
# Localización de los ficheros originales
# --------------------------------------------------------------------------
def localizar_directorio_gzip() -> Path | None:
    """Busca el directorio con los .gz originales de EMNIST."""
    candidatos = sorted(
        (RAIZ_TFDS / "downloads" / "extracted").glob("**/emnist-letters-train-images-idx3-ubyte.gz")
    )
    for ruta in candidatos:
        if ruta.stat().st_size > 0:
            return ruta.parent
    return None


def asegurar_descarga() -> Path:
    """Devuelve el directorio con los .gz, descargando EMNIST si hace falta."""
    directorio = localizar_directorio_gzip()
    if directorio is not None:
        return directorio

    print("No se encontraron los ficheros originales de EMNIST; descargando...")
    import tensorflow_datasets as tfds

    tfds.builder("emnist/letters").download_and_prepare()

    directorio = localizar_directorio_gzip()
    if directorio is None:
        raise FileNotFoundError(
            "No se pudieron localizar los ficheros idx de EMNIST/letters tras la descarga. "
            f"Revisa {RAIZ_TFDS / 'downloads' / 'extracted'}."
        )
    return directorio


# --------------------------------------------------------------------------
# Lectura del formato IDX
# --------------------------------------------------------------------------
def _leer_idx1(ruta: Path) -> np.ndarray:
    with gzip.open(ruta, "rb") as fichero:
        datos = fichero.read()
    magic = int.from_bytes(datos[0:4], "big")
    if magic != 2049:
        raise ValueError(f"{ruta.name}: magic idx1 inesperado ({magic}), se esperaba 2049.")
    n = int.from_bytes(datos[4:8], "big")
    return np.frombuffer(datos, dtype=np.uint8, count=n, offset=8).copy()


def _leer_idx3(ruta: Path) -> np.ndarray:
    with gzip.open(ruta, "rb") as fichero:
        datos = fichero.read()
    magic = int.from_bytes(datos[0:4], "big")
    if magic != 2051:
        raise ValueError(f"{ruta.name}: magic idx3 inesperado ({magic}), se esperaba 2051.")
    n = int.from_bytes(datos[4:8], "big")
    alto = int.from_bytes(datos[8:12], "big")
    ancho = int.from_bytes(datos[12:16], "big")
    plano = np.frombuffer(datos, dtype=np.uint8, count=n * alto * ancho, offset=16)
    return plano.reshape(n, alto, ancho).copy()


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------
def cargar_split(split: str, directorio: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Carga un split completo de EMNIST/letters.

    Args:
        split: 'train' o 'test'.

    Returns:
        (imagenes, etiquetas) donde imagenes es uint8 de forma (N, 28, 28) ya
        con la orientación corregida, y etiquetas es int32 en el rango 0..25.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split debe ser 'train' o 'test', no {split!r}")

    directorio = directorio or asegurar_descarga()
    imagenes = _leer_idx3(directorio / PATRON.format(split=split, tipo="images", n=3))
    etiquetas = _leer_idx1(directorio / PATRON.format(split=split, tipo="labels", n=1))

    if len(imagenes) != len(etiquetas):
        raise ValueError(
            f"{split}: {len(imagenes)} imágenes frente a {len(etiquetas)} etiquetas."
        )

    # EMNIST guarda las imágenes traspuestas (giradas 90° y espejadas).
    # Intercambiamos alto y ancho para dejarlas derechas.
    imagenes = np.transpose(imagenes, (0, 2, 1))

    # Las etiquetas van de 1 a 26; las llevamos a 0..25.
    etiquetas = etiquetas.astype(np.int32) - DESPLAZAMIENTO_ETIQUETA

    if imagenes.shape[1:] != (ALTO, ANCHO):
        raise ValueError(f"{split}: forma inesperada {imagenes.shape[1:]}, se esperaba {(ALTO, ANCHO)}")
    if etiquetas.min() < 0 or etiquetas.max() >= NUM_CLASES:
        raise ValueError(
            f"{split}: etiquetas fuera de rango [{etiquetas.min()}, {etiquetas.max()}]."
        )

    return imagenes, etiquetas


def resumen(split: str, etiquetas: np.ndarray) -> str:
    conteo = np.bincount(etiquetas, minlength=NUM_CLASES)
    return (
        f"{split}: {len(etiquetas):,} ejemplos | {int((conteo > 0).sum())}/{NUM_CLASES} clases "
        f"| min {conteo.min()} / max {conteo.max()} por clase"
    )


if __name__ == "__main__":
    directorio = asegurar_descarga()
    print(f"Ficheros originales en: {directorio}\n")
    for split in ("train", "test"):
        x, y = cargar_split(split, directorio)
        print(resumen(split, y))
