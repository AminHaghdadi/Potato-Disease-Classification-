from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

origins = [
    'http://localhost',
    'http://localhost3000'
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model('../saved_models/1')
class_name = ["Early Blight", "Late Blight", "Healthy"]


def image_to_array(file):
    image = np.array(Image.open(BytesIO(file)))
    return image


@app.post('/prediction')
async def preddiction(file: UploadFile = File(...)):
    img = image_to_array(await file.read())
    img = np.expand_dims(img, 0)
    predict = MODEL.predict(img)
    predict = predict[0]
    prediction = class_name[np.argmax(predict)]
    confidence = np.max(predict) * 100
    return {'Class': prediction,
            'confidence': confidence
            }


if __name__ == '__main__':
    uvicorn.run(app, port=8001, host='localhost')
