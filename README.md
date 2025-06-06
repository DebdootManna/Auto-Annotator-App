# Auto-Annotator App

A Python application that performs automatic annotation of images using a pre-trained YOLOv8 model.

## Features

- Download and train a YOLOv8 model on a custom dataset
- Generate annotations for unlabelled images
- Web-based GUI for uploading images and visualizing bounding boxes
- Export annotations in YOLO format

## Project Structure

```
auto_annotator/
├── dataset_annotated/           # Already downloaded
│   ├── images/                  # Images for training
│   └── labels/                  # YOLO format labels
├── dataset_unlabelled/          # Unlabelled images to annotate
├── model/                       # Stores model files
│   └── best.pt                  # Trained model file
├── scripts/                     # Processing scripts
│   ├── 1_download_model.py
│   ├── 2_train_model.py
│   └── 3_generate_annotations.py
├── app/                         # Web application
│   ├── main.py                  # Flask app
│   ├── static/
│   └── templates/
│       └── index.html
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/auto-annotator.git
   cd auto-annotator
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**
   - Place your annotated dataset in `dataset_annotated/` with images in `dataset_annotated/images/` and labels in `dataset_annotated/labels/`
   - Place your unlabelled images in `dataset_unlabelled/`

## Usage

### 1. Download the base model

```bash
python scripts/1_download_model.py
```

This will download the YOLOv8n model and save it to `model/yolov8n.pt`.

### 2. Train the model

```bash
python scripts/2_train_model.py --epochs 100 --img-size 640 --batch-size 16
```

This will train the model on your annotated dataset and save the best weights to `model/best.pt`.

### 3. Generate annotations for a batch of images

```bash
python scripts/3_generate_annotations.py --conf 0.25 --batch-size 16
```

This will use the trained model to generate annotations for your unlabelled images and save them to `generated_labels/`.

### 4. Run the web app

```bash
cd app
python main.py
```

Then open your browser and go to http://localhost:5000 to use the web app.

## Web App Usage

1. Drag and drop an image or click "Browse Files" to upload an image
2. The app will display the image with bounding boxes and show the YOLO format annotation
3. Click "Export Annotations" to download a ZIP file containing all the annotation files

## YOLO Annotation Format

The annotations are in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Integer class ID (0-based)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized width and height (0-1)

## Next Steps

The complete Auto-Annotator app is now ready! Here's a summary of what each component does:

1. **Download Model Script**: Downloads a pre-trained YOLOv8 model to use as a starting point
2. **Training Script**: Fine-tunes the model on your small labeled dataset
3. **Annotation Generator**: Uses the trained model to annotate new images in batch mode
4. **Web Application**: Provides a user-friendly interface for uploading images and getting annotations

To run the application:

1. Install the requirements with `pip install -r requirements.txt`
2. Run the scripts in sequence (1, 2, 3)
3. Start the web app with `python app/main.py`
4. Access the interface at http://localhost:5000

The app will generate annotation files in YOLO format that can be used for further training or data management.