"""
Servidor de predicción de letras.

Levanta un servidor web que sirve la página `web/index.html` (un recuadro
donde dibujar una letra) y expone el endpoint que recibe la imagen, la
preprocesa con exactamente el mismo pipeline que se usó con las imágenes
propias y devuelve la letra predicha con su probabilidad.

Endpoints:
    GET  /            página HTML con el recuadro de dibujo
    GET  /estado      métricas del modelo cargado
    POST /predecir    recibe la imagen y devuelve la predicción

    El POST admite dos formatos:
      - JSON:      {"imagen": "data:image/png;base64,...."}
      - multipart: campo de archivo llamado "imagen"

    Respuesta JSON:
      {
        "letra": "A",
        "probabilidad": 0.97,
        "top5": [["A", 0.97], ["H", 0.01], ...],
        "se_invirtio": true,
        "imagen_procesada": "data:image/png;base64,..."   # lo que ve la red
      }

Uso:
    python src/server.py                 # http://127.0.0.1:5000
    python src/server.py --puerto 8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from config import DIR_WEB, RUTA_METRICAS, RUTA_MODELO, indice_a_letra
from preprocess import preparar_para_modelo, preprocesar_imagen

app = Flask(__name__, static_folder=None)

_modelo = None  # se carga una sola vez, de forma perezosa


@app.after_request
def permitir_origen_cruzado(respuesta):
    """Cabeceras CORS abiertas, pensadas para desarrollo local.

    Lo habitual es abrir la página servida por este mismo servidor, en cuyo
    caso no hacen falta. Se incluyen para que la página siga funcionando si
    se abre desde otro origen (por ejemplo la extensión Live Server de VS
    Code o el archivo directamente desde el disco).
    """
    respuesta.headers["Access-Control-Allow-Origin"] = "*"
    respuesta.headers["Access-Control-Allow-Headers"] = "Content-Type"
    respuesta.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return respuesta


def obtener_modelo():
    global _modelo
    if _modelo is None:
        import tensorflow as tf

        if not RUTA_MODELO.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {RUTA_MODELO}. "
                "Entrena primero con: python src/train_model.py"
            )
        print(f"Cargando modelo desde {RUTA_MODELO} ...")
        _modelo = tf.keras.models.load_model(RUTA_MODELO)
        print("Modelo cargado.")
    return _modelo


def _imagen_desde_peticion():
    """Extrae la imagen de la petición, ya sea JSON en base64 o multipart."""
    if request.files.get("imagen"):
        return request.files["imagen"].read()

    datos = request.get_json(silent=True) or {}
    cadena = datos.get("imagen")
    if not cadena:
        return None
    if "," in cadena and cadena.strip().startswith("data:"):
        cadena = cadena.split(",", 1)[1]
    try:
        return base64.b64decode(cadena)
    except Exception:
        return None


def _a_data_url(arreglo_28x28: np.ndarray) -> str:
    """Convierte el arreglo normalizado en un PNG base64 para mostrarlo."""
    img = Image.fromarray((np.clip(arreglo_28x28, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((140, 140), Image.NEAREST)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
@app.get("/")
def pagina_principal():
    return send_from_directory(DIR_WEB, "index.html")


@app.get("/estado")
def estado():
    info = {"modelo_cargado": RUTA_MODELO.exists(), "ruta_modelo": str(RUTA_MODELO)}
    if RUTA_METRICAS.exists():
        metricas = json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
        info.update(
            {
                "arquitectura": metricas.get("arquitectura"),
                "exactitud_test": metricas.get("exactitud_test"),
                "dataset": metricas.get("dataset"),
            }
        )
    return jsonify(info)


@app.post("/predecir")
def predecir():
    crudo = _imagen_desde_peticion()
    if not crudo:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    try:
        arreglo, pasos = preprocesar_imagen(crudo, centrar=True, devolver_pasos=True)
    except Exception as exc:
        return jsonify({"error": f"No se pudo procesar la imagen: {exc}"}), 400

    # Un lienzo en blanco no debe inventar una predicción.
    if float(arreglo.max()) <= 0.05:
        return jsonify({"error": "La imagen está vacía: dibuja una letra."}), 400

    try:
        modelo = obtener_modelo()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    probas = modelo.predict(preparar_para_modelo(arreglo), verbose=0)[0]
    indice = int(np.argmax(probas))
    orden = np.argsort(probas)[::-1][:5]

    return jsonify(
        {
            "letra": indice_a_letra(indice),
            "probabilidad": float(probas[indice]),
            "top5": [[indice_a_letra(i), float(probas[i])] for i in orden],
            "se_invirtio": bool(pasos["se_invirtio"]),
            "imagen_procesada": _a_data_url(arreglo),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Servidor de predicción de letras")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--puerto", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--precargar",
        action="store_true",
        help="carga el modelo al arrancar en lugar de en la primera petición",
    )
    args = parser.parse_args()

    if args.precargar:
        obtener_modelo()

    print(f"Servidor escuchando en http://{args.host}:{args.puerto}")
    app.run(host=args.host, port=args.puerto, debug=args.debug)


if __name__ == "__main__":
    main()
