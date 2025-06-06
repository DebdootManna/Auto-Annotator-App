#!/usr/bin/env python3
"""
Script to download a pre-trained YOLOv8 model.
"""

import os
from ultralytics import YOLO

def download_model(model_name="yolov8n.pt", save_dir="model"):
    """
    Download a pre-trained YOLOv8 model.

    Args:
        model_name (str): Name of the model to download (default: yolov8n.pt)
        save_dir (str): Directory to save the model (default: model)
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Check if the model already exists
    save_path = os.path.join(save_dir, model_name)
    if os.path.exists(save_path):
        print(f"Model {model_name} already exists at {save_path}")
        return save_path

    # Download the model
    print(f"Downloading model {model_name}...")
    model = YOLO(model_name)

    # Save the model
    print(f"Model downloaded and saved to {save_path}")
    return save_path

if __name__ == "__main__":
    download_model()
