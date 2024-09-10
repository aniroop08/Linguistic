from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import cv2
from shutil import rmtree

np.random.seed(42)
tf.random.set_seed(42)

model = load_model('ocr\\models\\handwriting_recognition.h5')

characters = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def preprocessing(image_path):
    img = cv2.imread(image_path)

    def getSkewAngle(cvImage) -> float:
        newImage = cvImage.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    def rotateImage(cvImage, angle: float):
        newImage = cvImage.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return newImage

    angle = getSkewAngle(img)
    img = rotateImage(img, -1.0 * angle)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape

    if w > 1000:

        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)

        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)

    def thresholding(image):
        img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
        return thresh

    def dilated(thresh_img):
        kernel = np.ones((3,85), np.uint8)
        dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
        return dilated

    (contours, heirarchy) = cv2.findContours(dilated(thresholding(img)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1])

    img2 = img.copy()
    i = 0
    for ctr in sorted_contours_lines:

        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 2)
        cv2.putText(img2, f"{i}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 2)
        i += 1

    try:
        os.mkdir('temp')
        os.mkdir('temp\\temp')
    except FileExistsError:
        pass

    sentences = []
    i = 0
    for ctr in sorted_contours_lines:
        x,y,w,h = cv2.boundingRect(ctr)

        image = img[y:y+h, x:x+w]

        cv2.imwrite(f"temp\\temp\\{i}.jpg", image)
        sentences.append(f'temp\\temp\\{i}.jpg')
        i += 1

    def noise_removal(image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)

    def thin_font(image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)

    image_paths = []
    k = 0
    for i in sentences:
        img2 = cv2.imread(i)
        kernel = np.ones((3,15), np.uint8)
        dilated2 = cv2.dilate(thresholding(img2), kernel, iterations = 1)
        (cnt, heirarchy) = cv2.findContours(dilated2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])

        for word in sorted_contour_words:
            if cv2.contourArea(word) < 400:
                continue

            x, y, w, h = cv2.boundingRect(word)

            image = img2[y:y+h, x:x+w]
            image = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
            (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            image = noise_removal(image)
            image = thin_font(image)

            cv2.imwrite(f"temp\\{k}.jpg", image)
            image_paths.append(f'temp\\{k}.jpg')
            k += 1

    return image_paths

AUTOTUNE = tf.data.AUTOTUNE

char_to_num = StringLookup(vocabulary = characters, mask_token = None)

num_to_char = StringLookup(vocabulary = char_to_num.get_vocabulary(), mask_token = None, invert = True)

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(image,paddings=[[pad_height_top, pad_height_bottom],[pad_width_left, pad_width_right],[0, 0]])

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

image_width = 128
image_height = 32
def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def prepare_dataset(image_path):
    batch = 64
    dataset = tf.data.Dataset.from_tensor_slices(image_path).map(preprocess_image, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch).cache().prefetch(AUTOTUNE)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = ""
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text += res + " "
    return output_text

def detect(image_path):
    image_paths = preprocessing(image_path)
    image_ds = prepare_dataset(image_paths)
    for batch in image_ds.take(1):
        preds = model.predict(batch)
        pred_texts = decode_batch_predictions(preds)
    try:
        rmtree('temp')
    except FileNotFoundError:
        pass

    return pred_texts[:]