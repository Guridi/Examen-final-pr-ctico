"""
Configuración central del proyecto.

Todas las rutas y constantes compartidas viven aquí para que los demás
módulos (entrenamiento, preprocesamiento, servidor, reporte) usen
exactamente los mismos valores.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent

DIR_MODELO = RAIZ / "modelo"
DIR_IMAGENES_PROPIAS = RAIZ / "imagenes_propias"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_WEB = RAIZ / "web"

RUTA_MODELO = DIR_MODELO / "modelo_emnist.keras"
RUTA_METRICAS = DIR_MODELO / "metricas.json"
RUTA_HISTORIAL = DIR_MODELO / "historial.json"

for _d in (DIR_MODELO, DIR_IMAGENES_PROPIAS, DIR_RESULTADOS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset EMNIST
# --------------------------------------------------------------------------
# Usamos la variante "letters" de EMNIST: 26 clases balanceadas (A-Z),
# donde mayúsculas y minúsculas de la misma letra comparten etiqueta.
# https://www.tensorflow.org/datasets/catalog/emnist
NOMBRE_DATASET = "emnist/letters"

# EMNIST/letters entrega las etiquetas en el rango 1..26. Restamos 1 para
# trabajar con índices 0..25 que es lo que espera la capa softmax.
DESPLAZAMIENTO_ETIQUETA = 1

CLASES = [chr(ord("A") + i) for i in range(26)]  # ['A', 'B', ..., 'Z']
NUM_CLASES = len(CLASES)

# --------------------------------------------------------------------------
# Formato de imagen
# --------------------------------------------------------------------------
ALTO = 28
ANCHO = 28
CANALES = 1
FORMA_ENTRADA = (ALTO, ANCHO, CANALES)

# EMNIST normaliza cada carácter dentro de una caja de 20x20 y luego lo
# centra por su centro de masa en un lienzo de 28x28. Replicamos ese mismo
# criterio al preprocesar nuestras propias imágenes.
CAJA_INTERNA = 20

# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
TAM_LOTE = 128
EPOCAS = 20
SEMILLA = 42
FRACCION_VALIDACION = 0.1  # porción del split de entrenamiento usada para validar


def indice_a_letra(indice: int) -> str:
    """Convierte un índice de clase (0..25) en su letra correspondiente."""
    return CLASES[int(indice)]
