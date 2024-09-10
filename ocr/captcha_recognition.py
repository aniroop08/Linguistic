import numpy as np

from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = load_model('ocr\\models\\captcha_recognition.h5')

characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
              'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
              'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

AUTOTUNE = tf.data.AUTOTUNE

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


def preprocess_image(img_path):
    img_width = 200
    img_height = 50
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return img

def process_images(image_path):
    image = preprocess_image(image_path)
    return image

def prepare_dataset(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(process_images, num_parallel_calls=AUTOTUNE)
    return dataset.batch(1).cache().prefetch(AUTOTUNE)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    result = ''
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        res = ''
        for i in result:
            if i == '[':
                break
            res += i
    return res

def detect(image_path):
    image_ds = prepare_dataset(image_path)
    preds = model.predict(image_ds.take(1))
    pred_texts = decode_batch_predictions(preds)
    return pred_texts