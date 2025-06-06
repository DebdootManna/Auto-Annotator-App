#!/usr/bin/env python3
"""
Script to train a YOLOv8 model on a custom dataset.
"""

import os
import yaml
import shutil
import argparse
from ultralytics import YOLO

def create_data_yaml(train_path, val_path, class_names, yaml_path="data.yaml"):
    """
    Create a data.yaml file for training.

    Args:
        train_path (str): Path to training data
        val_path (str): Path to validation data
        class_names (list): List of class names
        yaml_path (str): Path to save the yaml file
    """
    data = {
        'train': train_path,
        'val': val_path,
        'nc': len(class_names),
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Created data.yaml at {yaml_path}")
    return yaml_path

def train_model(model_path, data_yaml, epochs=100, img_size=640, batch_size=16, save_dir="model"):
    """
    Train a YOLOv8 model on a custom dataset.

    Args:
        model_path (str): Path to the model
        data_yaml (str): Path to the data.yaml file
        epochs (int): Number of epochs to train for
        img_size (int): Image size for training
        batch_size (int): Batch size for training
        save_dir (str): Directory to save the model
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load the model
    model = YOLO(model_path)

    # Train the model with MPS device
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=save_dir,
        name="train",
        exist_ok=True,
        device='mps',  # Use MPS for M2 chip
        workers=4,     # Optimize for M2
        cache=True,    # Enable caching for faster training
        amp=True       # Enable automatic mixed precision
    )

    # Copy the best weights to model/best.pt
    best_model_path = os.path.join(save_dir, "train", "weights", "best.pt")
    if os.path.exists(best_model_path):
        target_path = os.path.join(save_dir, "best.pt")
        shutil.copy(best_model_path, target_path)
        print(f"Best model saved to {target_path}")
    else:
        print(f"Warning: Best model not found at {best_model_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model on a custom dataset.")
    parser.add_argument("--model", type=str, default="model/yolov8l.pt", help="Path to the model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()

    # Create data.yaml with correct paths
    data_yaml = create_data_yaml(
        train_path="dataset_annotated/images",
        val_path="dataset_annotated/images",
        class_names=["person", "helmet", "no helmet", "jumpsuit", "no jumpsuit"]
    )

    # Train the model
    train_model(
        model_path=args.model,
        data_yaml=data_yaml,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
