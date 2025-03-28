from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)


modelo = tf.keras.models.load_model("modelo.hdf5")


clases = {0: "Orgánico", 1: "Reciclable"}

@app.route("/")
def index():
    return render_template("index.html")  

@app.route("/predecir", methods=["POST"])
def predecir():
    try:
    
        data = request.json["imagen"]
        imagen_bytes = base64.b64decode(data.split(",")[1])  # Decodificar la imagen
        img = Image.open(io.BytesIO(imagen_bytes)).resize((180, 180))  # Redimensionar imagen
        img_array = np.array(img) / 255.0  # Normalizar la imagen
        img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensión para predicción

        # Hacer la predicción
        prediccion = modelo.predict(img_array)
        clase_predicha = int(np.argmax(prediccion)) 

        
        clase_nombre = clases.get(clase_predicha, "Desconocido")

        return jsonify({"clase": clase_nombre, "confianza": float(np.max(prediccion))})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
