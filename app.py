import os
from flask import Flask,render_template,request,session
from werkzeug.utils import secure_filename
from ocr import *
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/')
@app.route('/templates/index.html')
def home():
    return render_template("index.html")


# Text Detection


@app.route('/templates/text_det.html')
def text_det_home():
    return render_template('text_det.html')

@app.route('/templates/text_det.html/',methods=('POST','GET'))
def text_det_output():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        text = text_recognition.get_Text(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        if text=="":
            text = "No text found"
 
        return render_template('text_det2.html',user_image='../../'+session['uploaded_img_file_path'],output=text)


# Captcha Detection


@app.route('/templates/captcha_det.html')
def captcha_det_home():
    return render_template('captcha_det.html')

@app.route('/templates/captcha_det.html/',methods=('POST','GET'))
def captcha_det_output():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        text = captcha_recognition.detect([os.path.join(app.config['UPLOAD_FOLDER'], img_filename)])
        if text=="":
            text = "No text found"
 
        return render_template('captcha_det2.html',user_image='../../'+session['uploaded_img_file_path'],output=text)


# Handwriting Detection


@app.route('/templates/hand_det.html')
def hand_det_home():
    return render_template('hand_det.html')

@app.route('/templates/hand_det.html/',methods=('POST','GET'))
def hand_det_output():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        text = handwriting_recognition.detect(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        if text=="":
            text = "No text found"
 
        return render_template('hand_det2.html',user_image='../../'+session['uploaded_img_file_path'],output=text)

# Language Translation


@app.route('/templates/lang_trans.html')
def lang_trans_home():
    return render_template('lang_trans.html')

@app.route('/templates/lang_trans.html',methods=("POST","GET"))
def lang_trans_out():
    if request.method=='POST':
        target_lang = request.form.get('Target')
        input = request.form.get('Input')
        output = translator.translate(input, target_lang)

        return render_template('lang_trans.html', output = output)


# Number Plate Detection

@app.route('/templates/num_plate_det.html')
def num_plate_det_home():
    return render_template('num_plate_det.html')

@app.route('/templates/num_plate_det.html/',methods=('POST','GET'))
def num_plate_det_output():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        img, text = license_plate_recognition.detect(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], img_filename), img)
        if text=="":
            text = "No text found"
 
        return render_template('num_plate_det2.html',user_image='../../'+session['uploaded_img_file_path'],output=text)


if __name__ == "__main__":
    app.run(debug = True)