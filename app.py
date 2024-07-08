# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:01:13 2024

@author: Laith Qushair
"""

import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import os
import matplotlib.image as mpimg
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = 'images'

model = tf.keras.models.load_model('models/skin_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

class_names = ['Actinic keratoses and intraepithelial carcinoma / Bowens disease', 'basal cell carcinoma',
               'benign keratosis-like lesions', 'dermatofibroma ', 'melanoma ','melanocytic nevi', 'vascular lesions']
@app.route('/predict',methods=["POST","GET"])
def predict():
    image = request.files['file']
    #if image.filename == "":
     #   print("File name is invalid")
      #  return redirect(request.url)
    #int_features = [x for x in request.form.values()] #Convert string inputs to float.
    #img = mpimg.imread(int_features[0])
    #features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    #prediction = model.predict(features)  # features Must be in the form [[a, b]]
    filename = secure_filename(image.filename)
    basedir = os.path.abspath(os.path.dirname(__file__))
    image.save(os.path.join(basedir,app.config['IMAGE_UPLOADS'],filename))
    img = mpimg.imread(os.path.join(basedir,app.config['IMAGE_UPLOADS'],filename))
    res = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    x = np.expand_dims(res, axis=0)
    results = model.predict(x)
    highest = 0
    for i in range(len(results[0])):
        if results[0][i] > highest:
            highest = i
    os.remove(os.path.join(basedir,app.config['IMAGE_UPLOADS'],filename))
    #return render_template("index.html",prediction_text='The skin abnormality is most likely {}'.format(class_names[highest]))
    return render_template("index.html",prediction_text='The model predicts this skin abnormality is {}'.format(class_names[highest]))
    


if __name__ == "__main__":
    app.run(port=5000)
