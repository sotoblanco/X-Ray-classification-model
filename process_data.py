import tflite_runtime.interpreter as tflite

import os
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

MODEL_NAME = os.getenv('MODEL_NAME', 'pneumonia-class.tflite')

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255.0


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return float(preds[0, 0])


#def lambda_handler(event, context):
#    url = event['url']
#    pred = predict(url)
#    result = np.where(pred > 0.8, 'Pneumonia', 'Normal')

 #   return {'prediction':result}


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    if pred > 0.8: 
        pred_string = "Pneumonia"
    else:
        pred_string = "Normal"
        
    result = {
        'prediction': pred_string
    }

    return result
event = {'url':"https://user-images.githubusercontent.com/46135649/207101011-77379ccc-6684-4b74-93e4-852367e28920.png" }
print(lambda_handler(event, None))