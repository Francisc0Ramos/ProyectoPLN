import easyocr
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import pinyin


reader = easyocr.Reader(['ch_sim', 'en'])
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

model_name_en_es = "Helsinki-NLP/opus-mt-en-es"
tokenizer_en_es = MarianTokenizer.from_pretrained(model_name_en_es)
model_en_es = MarianMTModel.from_pretrained(model_name_en_es)

def reconocerTexto(image):
  img_array = np.array(image)
  result = reader.readtext(img_array)
  caracteres = [detection[1] for detection in result]
  return caracteres

def obtenerPinyin(texto):
    texto_concatenado = ''.join(texto)
    pinyin_texto = pinyin.get(texto_concatenado)
    return pinyin_texto

def traducirTexto(texto):
  inputs = tokenizer(texto, return_tensors="pt",padding=True,truncation = True)
  translated = model.generate(**inputs)
  translated_text = tokenizer.decode(translated[0],skip_special_tokens=True)
  return translated_text

def traducirTextoEnEs(texto):
  inputs = tokenizer_en_es(texto, return_tensors="pt",padding=True,truncation = True)
  translated = model_en_es.generate(**inputs)
  translated_text = tokenizer_en_es.decode(translated[0],skip_special_tokens=True)
  return translated_text

from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/subir_imagen": {"origins": "*"}})

@app.route('/subir_imagen', methods=['POST'])
def upload_image():
    # Obtener el dataURL enviado desde el cliente
    data = request.get_json()
    image_data = data['image']
    
    # Eliminar el prefijo "data:image/jpeg;base64," que está en el dataURL
    image_data = image_data.split(",")[1]
    
    # Convertir el string base64 a datos binarios
    img_data = base64.b64decode(image_data)
    
    # Crear una imagen desde los datos binarios (opcional: guardar la imagen como archivo)
    image = Image.open(BytesIO(img_data))

    textoChino = reconocerTexto(image)
    textoPinyin = obtenerPinyin(textoChino)
    textoIngles = traducirTexto(textoChino)
    textoEsp = traducirTextoEnEs(textoIngles)

    return jsonify({"textoChino": textoChino, "textoPinyin": textoPinyin, "textoIngles": textoIngles, "textoEsp": textoEsp})
    

if __name__ == '__main__':
    app.run(debug=True)
