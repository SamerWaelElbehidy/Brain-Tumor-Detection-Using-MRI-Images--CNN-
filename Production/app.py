# pylint: skip-file
import os
from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model

UPLOADS_FOLDER = '/static/uploads/'
model = None
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home_page():
    '''
    The browser will render home.html when it visits '/' (the root of the web app)
    '''
    return render_template('home.html')


@app.route('/model', methods=['GET'])
def models_page():
    '''
    The browser will render model.html when it visits '/model'
    '''
    global model
    model = load_model('./tetabyte.keras')
    return render_template('model.html')


@app.route('/model', methods=['POST'])
def model_page():
    '''
    Defines what the browser should do when a post request (e.g. upload) is done on /model 
    '''
    if request.method == 'POST':
        
        # We can't proceed if the post request doesn't have the file or the file was not selected.
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('model.html', err_msg='No File Selected!')
        
        # Okay, so we have it
        file = request.files['file']
        # You can check if the filetype is correct here, we skippped that for simplicity.
        path = os.path.join(os.getcwd() + UPLOADS_FOLDER, file.filename)
        # Save the file in the uploads folder so the 
        file.save(path)

        # call the model on it
        model_output = predict_image(path)

        #display the model's output
        return render_template('model.html', err_msg='', model_output=model_output)

def predict_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Preprocess the image
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB            # [L,W]
    img_array = img.astype('float32') / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension [1, L, W]

    # Make predictions
    prob = model.predict(img_array)[0][0]
    return "Tumor Detected" if prob > 0.5 else "Healthy"

if __name__ == '__main__':
    app.run(debug=True, port=5001)