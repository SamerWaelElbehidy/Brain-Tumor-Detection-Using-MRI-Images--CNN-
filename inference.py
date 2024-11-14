import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('./tetabyte.keras')

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

# Example usage
result = predict_image('dataset\yes\Y249.JPG')
print('Prediction:', result)