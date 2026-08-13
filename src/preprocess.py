"""
Preprocesamiento de imágenes propias para que coincidan con el formato EMNIST.

El proceso implementa los cuatro pasos exigidos:

    1. Conversión a escala de grises.
    2. Redimensionamiento a 28 x 28 píxeles.
    3. Normalización de los valores de los píxeles al rango [0, 1].
    4. Inversión de colores si fuese necesario (EMNIST usa trazo blanco
       sobre fondo negro).

Además, de forma opcional (activada por defecto) se replica el criterio de
normalización geométrica del propio EMNIST/MNIST: el carácter se recorta,
se escala para caber en una caja de 20x20 y se centra por su centro de masa
dentro del lienzo de 28x28. Este paso extra es el que permite que una foto o
un trazo de canvas se parezca de verdad a una muestra del dataset.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from config import ALTO, ANCHO, CAJA_INTERNA, CLASES

try:  # scipy da un centrado subpíxel más limpio, pero no es imprescindible
    from scipy.ndimage import shift as _desplazar_subpixel
except ImportError:  # pragma: no cover
    _desplazar_subpixel = None


# --------------------------------------------------------------------------
# Utilidades internas
# --------------------------------------------------------------------------
def _a_imagen_pil(origen) -> Image.Image:
    """Acepta ruta, bytes, objeto PIL o arreglo numpy y devuelve una imagen PIL."""
    if isinstance(origen, Image.Image):
        return origen
    if isinstance(origen, (str, Path)):
        return Image.open(origen)
    if isinstance(origen, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(origen)))
    if isinstance(origen, np.ndarray):
        arr = origen
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Tipo de imagen no soportado: {type(origen)!r}")


def _aplanar_transparencia(img: Image.Image) -> Image.Image:
    """Compone sobre fondo blanco si la imagen trae canal alfa.

    Sin esto, el PNG que envía el canvas del navegador (fondo transparente)
    se convertiría en negro y el detector de inversión fallaría.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fondo, img)
    return img


def _hay_que_invertir(gris: np.ndarray) -> bool:
    """Decide si la imagen necesita inversión de colores.

    EMNIST guarda trazo claro sobre fondo oscuro. Si el borde de la imagen es
    más claro que su interior, estamos ante tinta oscura sobre papel blanco y
    hay que invertir.
    """
    borde = np.concatenate([gris[0, :], gris[-1, :], gris[:, 0], gris[:, -1]])
    return float(np.median(borde)) > 127.0


def _recortar_a_la_tinta(gris: np.ndarray, umbral: int = 30) -> np.ndarray:
    """Recorta la imagen a la caja mínima que contiene el trazo."""
    mascara = gris > umbral
    if not mascara.any():
        return gris  # imagen en blanco: no hay nada que recortar
    filas = np.where(mascara.any(axis=1))[0]
    cols = np.where(mascara.any(axis=0))[0]
    return gris[filas[0] : filas[-1] + 1, cols[0] : cols[-1] + 1]


def _centrar_por_centro_de_masa(lienzo: np.ndarray) -> np.ndarray:
    """Desplaza el contenido para que su centro de masa quede en el centro."""
    total = lienzo.sum()
    if total <= 0:
        return lienzo
    ys, xs = np.indices(lienzo.shape)
    cy = (ys * lienzo).sum() / total
    cx = (xs * lienzo).sum() / total
    dy = (lienzo.shape[0] - 1) / 2.0 - cy
    dx = (lienzo.shape[1] - 1) / 2.0 - cx
    if _desplazar_subpixel is not None:
        return _desplazar_subpixel(lienzo, (dy, dx), order=1, mode="constant", cval=0.0)
    return np.roll(np.roll(lienzo, int(round(dy)), axis=0), int(round(dx)), axis=1)


# --------------------------------------------------------------------------
# Preprocesamiento principal
# --------------------------------------------------------------------------
def preprocesar_imagen(origen, centrar: bool = True, devolver_pasos: bool = False):
    """Convierte una imagen cualquiera en un arreglo 28x28 listo para el modelo.

    Args:
        origen: ruta, bytes, imagen PIL o arreglo numpy.
        centrar: si True aplica la normalización geométrica estilo EMNIST
            (recorte + caja de 20x20 + centrado por centro de masa).
            Si False aplica el redimensionado directo a 28x28.
        devolver_pasos: si True devuelve además un diccionario con las
            imágenes intermedias, útil para ilustrar el proceso.

    Returns:
        np.ndarray de forma (28, 28) y valores float32 en [0, 1].
        Si devolver_pasos=True, una tupla (arreglo, pasos).
    """
    pasos = {}
    img = _aplanar_transparencia(_a_imagen_pil(origen))
    pasos["original"] = img.convert("RGB")

    # --- PASO 1: conversión a escala de grises -----------------------------
    gris = np.asarray(img.convert("L"), dtype=np.float32)
    pasos["gris"] = gris.copy()

    # --- PASO 4 (aplicado aquí): inversión de colores si es necesario ------
    # Se resuelve antes del recorte porque todo lo que sigue asume el
    # convenio de EMNIST: trazo claro (valores altos) sobre fondo oscuro.
    invertida = _hay_que_invertir(gris)
    if invertida:
        gris = 255.0 - gris
    pasos["invertida"] = gris.copy()
    pasos["se_invirtio"] = invertida

    # Limpieza suave: llevamos el fondo real a cero y reescalamos el contraste
    # para que el trazo llegue a 255. Esto elimina el ruido de compresión de
    # fotos y evita que un fondo gris se confunda con tinta.
    fondo = float(np.percentile(gris, 50))
    gris = np.clip(gris - fondo, 0, None)
    maximo = float(gris.max())
    if maximo > 0:
        gris = gris * (255.0 / maximo)

    if centrar:
        # --- PASO 2: redimensionamiento a 28x28 (con criterio EMNIST) ------
        recorte = _recortar_a_la_tinta(gris)
        alto, ancho = recorte.shape
        if alto == 0 or ancho == 0 or recorte.max() <= 0:
            final = np.zeros((ALTO, ANCHO), dtype=np.float32)
        else:
            # Escalamos el lado mayor a 20 px conservando la relación de aspecto.
            escala = CAJA_INTERNA / float(max(alto, ancho))
            nuevo_alto = max(1, int(round(alto * escala)))
            nuevo_ancho = max(1, int(round(ancho * escala)))
            reducida = Image.fromarray(recorte.astype(np.uint8)).resize(
                (nuevo_ancho, nuevo_alto), Image.LANCZOS
            )
            # Pegamos la caja de 20x20 centrada en el lienzo de 28x28.
            final = np.zeros((ALTO, ANCHO), dtype=np.float32)
            y0 = (ALTO - nuevo_alto) // 2
            x0 = (ANCHO - nuevo_ancho) // 2
            final[y0 : y0 + nuevo_alto, x0 : x0 + nuevo_ancho] = np.asarray(
                reducida, dtype=np.float32
            )
            # Ajuste fino por centro de masa, igual que hace MNIST/EMNIST.
            final = _centrar_por_centro_de_masa(final)
    else:
        # --- PASO 2 (variante simple): redimensionado directo a 28x28 ------
        final = np.asarray(
            Image.fromarray(gris.astype(np.uint8)).resize((ANCHO, ALTO), Image.LANCZOS),
            dtype=np.float32,
        )

    final = np.clip(final, 0, 255)
    pasos["redimensionada"] = final.copy()

    # --- PASO 3: normalización a [0, 1] ------------------------------------
    normalizada = (final / 255.0).astype(np.float32)
    pasos["normalizada"] = normalizada

    if devolver_pasos:
        return normalizada, pasos
    return normalizada


def preparar_para_modelo(arreglo_28x28: np.ndarray) -> np.ndarray:
    """Añade las dimensiones de lote y canal: (28,28) -> (1,28,28,1)."""
    return arreglo_28x28.reshape(1, ALTO, ANCHO, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Carga de las imágenes propias desde disco
# --------------------------------------------------------------------------
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def etiqueta_desde_nombre(ruta: Path) -> str | None:
    """Deduce la etiqueta real a partir del nombre del archivo.

    Convención aceptada: la etiqueta es el primer carácter alfabético del
    nombre, opcionalmente separado por '_' o '-'.  Ejemplos válidos:
        A_01.png -> 'A'      b-3.jpg -> 'B'      Z.png -> 'Z'
    """
    tallo = ruta.stem.strip()
    if not tallo:
        return None
    candidato = tallo.split("_")[0].split("-")[0].strip()
    if len(candidato) == 1 and candidato.isalpha():
        return candidato.upper()
    if tallo[0].isalpha():
        return tallo[0].upper()
    return None


def cargar_imagenes_propias(directorio: Path, centrar: bool = True) -> list[dict]:
    """Carga y preprocesa todas las imágenes del directorio indicado.

    Returns:
        Lista de diccionarios con las claves: ruta, nombre, etiqueta_real,
        arreglo (28x28 normalizado) y pasos (imágenes intermedias).
    """
    directorio = Path(directorio)
    archivos = sorted(
        p for p in directorio.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES
    )
    muestras = []
    for ruta in archivos:
        try:
            arreglo, pasos = preprocesar_imagen(ruta, centrar=centrar, devolver_pasos=True)
        except Exception as exc:  # una imagen corrupta no debe tumbar el lote
            print(f"  [aviso] no se pudo procesar {ruta.name}: {exc}")
            continue
        etiqueta = etiqueta_desde_nombre(ruta)
        if etiqueta is not None and etiqueta not in CLASES:
            etiqueta = None
        muestras.append(
            {
                "ruta": ruta,
                "nombre": ruta.name,
                "etiqueta_real": etiqueta,
                "arreglo": arreglo,
                "pasos": pasos,
            }
        )
    return muestras
