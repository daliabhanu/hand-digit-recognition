from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("path_to_your_trained_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the base64-encoded image data from the frontend
    image_data = request.values['imageData']
    # Convert base64-encoded image to a NumPy array
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1]))).convert('L')
    image = np.array(image.resize((28, 28)))
    # Flatten and normalize the image data
    image = image.reshape(1, -1) / 255.0
    # Make the prediction
    prediction = model.predict(image)[0]
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)
