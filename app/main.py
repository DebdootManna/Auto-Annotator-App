#!/usr/bin/env python3
"""
Flask web application for auto-annotation of images using a trained YOLOv8 model.
"""

import os
import io
import base64
import zipfile
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
MODEL_PATH = '../model/best.pt'
model = None

def load_model():
    """Load the YOLOv8 model from the specified path."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Model not found at {MODEL_PATH}. Using default YOLOv8n.")
            model = YOLO('yolov8n.pt')
    return model

def get_base64_image(img):
    """Convert an image to base64 string for display in HTML."""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def draw_bounding_boxes(image, results):
    """Draw bounding boxes on the image based on model results."""
    img_copy = image.copy()

    for box in results.boxes:
        # Get coordinates (convert to pixel values)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class ID and confidence
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        # Class name
        cls_name = results.names[cls_id]

        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label
        label = f"{cls_name} {conf:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img_copy, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_copy

def generate_yolo_annotation(results):
    """Generate YOLO format annotation from model results."""
    annotation_lines = []

    for box in results.boxes:
        # Get normalized coordinates
        x, y, w, h = box.xywhn[0].tolist()

        # Get class ID
        cls_id = int(box.cls.item())

        # Add to annotation lines
        annotation_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return "\n".join(annotation_lines)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and generate annotations."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to load image'})

        # Make a prediction
        model = load_model()
        results = model(img, verbose=False)[0]

        # Draw bounding boxes on the image
        img_with_boxes = draw_bounding_boxes(img, results)

        # Convert to base64 for display
        img_base64 = get_base64_image(img_with_boxes)

        # Generate YOLO annotation
        annotation = generate_yolo_annotation(results)

        return jsonify({
            'image': img_base64,
            'annotation': annotation,
            'filename': filename
        })

@app.route('/export', methods=['POST'])
def export_annotations():
    """Export annotations as a zip file."""
    data = request.json
    annotations = data.get('annotations', {})

    if not annotations:
        return jsonify({'error': 'No annotations to export'})

    # Create a temporary directory to store annotations
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create annotation files
        for filename, annotation in annotations.items():
            base_name = os.path.splitext(filename)[0]
            with open(os.path.join(temp_dir, f"{base_name}.txt"), 'w') as f:
                f.write(annotation)

        # Create a zip file
        zip_path = os.path.join(temp_dir, 'annotations.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.txt'):
                        zipf.write(os.path.join(root, file), file)

        # Send the zip file
        return send_file(zip_path, as_attachment=True, download_name='annotations.zip')

if __name__ == '__main__':
    app.run(debug=True)
