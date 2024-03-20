from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import uuid  # Import the uuid module to generate unique filenames

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load pre-trained MobileNetV2 model for image classification
model = MobileNetV2(weights='imagenet')

# Function to detect eyes in an image
def detect_eyes(image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if eye_cascade.empty():
        raise FileNotFoundError("Haar cascade file for eye detection not found.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return eyes

# Function to process uploaded image
def process_image(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    eyes1 = detect_eyes(img1)
    eyes2 = detect_eyes(img2)
    
    if len(eyes1) > 0 and len(eyes2) > 0:
        (x1, y1, w1, h1) = eyes1[0]
        (x2, y2, w2, h2) = eyes2[0]
        eye_region1 = img1[y1:y1+h1, x1:x1+w1]
        eye_region2 = img2[y2:y2+h2, x2:x2+w2]
        eye_region2 = cv2.resize(eye_region2, (w1, h1))
        img1[y1:y1+h1, x1:x1+w1] = eye_region2
        
        # Generate a unique filename using uuid
        output_filename = str(uuid.uuid4()) + '.jpg'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        cv2.imwrite(output_path, img1)
        return output_filename  # Return only the filename, not the full path
    else:
        return None

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return redirect(request.url)
        file1 = request.files['file1']
        file2 = request.files['file2']
        if file1.filename == '' or file2.filename == '':
            return redirect(request.url)
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = 'input.jpg'
            filename2 = 'reference.jpg'
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            output_filename = process_image(os.path.join(app.config['UPLOAD_FOLDER'], filename1),
                                            os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            if output_filename:
                return render_template('output.html', output_image=output_filename)
            else:
                return "Eyes not detected in one or both images."
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)