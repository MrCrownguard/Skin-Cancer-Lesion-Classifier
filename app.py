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

#Create an app object using the Flask class. 
app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = 'images'
#numbered_classes = dict(enumerate(class_names))

#Load the trained model. (Pickle file)
model = tf.keras.models.load_model('models/skin_model.keras')

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
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
    #folder = os.path.join(basedir,app.config['IMAGE_UPLOADS'])
    #for filename in os.listdir(folder):
    #    file_path = os.path.join(folder, filename)
    #    if os.path.isfile(file_path):
    #        os.remove(file_path)
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
    
#output = round(prediction[0], 2)

    #return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(img))


#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. clas
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run(port=5031)