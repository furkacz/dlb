# very simple api to interact with trained model
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette import status
from starlette.requests import Request
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import numpy as np

from utils import create_model, get_hparams, HP_IMAGE_SIZE


# setup model
labels = []
with open(f'api/labels.json', 'r') as f:
    labels = json.load(f)

mlb = MultiLabelBinarizer(classes=labels)
mlb.fit(labels)

hparams = {}
with open(f'api/hparams.json', 'r') as f:
    hconfig = json.load(f)
    hparams = get_hparams(hconfig['run'])

model = create_model(len(labels), hparams)
model.load_weights(f'api/model.keras')


# fastapi
app = FastAPI(
    title='dlb-api',
    description='Fashion products multi-label image classification',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/tag')
async def tag_image(request: Request):
    content_type = request.headers.get('content-type', None)
    if (content_type != 'image/jpeg'):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    
    image_bytes = b''
    async for chunk in request.stream():
        image_bytes += chunk
    
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [hparams[HP_IMAGE_SIZE], hparams[HP_IMAGE_SIZE]])
    img = tf.expand_dims(img, axis=0)

    result = np.array(np.round(model.predict(img)), np.float32)
    tags = list(mlb.inverse_transform(result)[0])
    return tags
