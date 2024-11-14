# pylint: skip-file
import os
from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sys
import io

# Set default encoding to UTF-8 for standard input/output
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

# Define absolute path for uploads folder
BASE_DIR = os.getcwd()  # Get the current working directory
UPLOADS_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)  # Create the folder if it doesn't exist

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
        
        # Check if the post request has a file
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('model.html', err_msg='No File Selected!')
        
        # Get the uploaded file
        file = request.files['file']
        
        # Ensure filename is properly encoded/decoded to avoid Unicode errors
        safe_filename = file.filename.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Create the full path to save the file in the uploads folder
        path = os.path.join(UPLOADS_FOLDER, safe_filename)
        
        # Save the file
        try:
            file.save(path)
        except Exception as e:
            return render_template('model.html', err_msg=f"Failed to save file: {e}")

        # Call the model to make a prediction on the uploaded image
        model_output = predict_image(path)

        # Display the model's output
        return render_template('model.html', err_msg='', model_output=model_output)

def predict_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Preprocess the image
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_array = img.astype('float32') / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension [1, L, W]

    # Make predictions
    prob = model.predict(img_array)[0][0]
    return "Tumor Detected" if prob > 0.5 else "Healthy"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
