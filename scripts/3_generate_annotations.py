#!/usr/bin/env python3
"""
Script to generate annotations for unlabelled images using a trained YOLOv8 model.
"""

import os
import glob
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def generate_annotations(model_path, images_dir, output_dir, conf_threshold=0.25, batch_size=16):
    """
    Generate annotations for unlabelled images.

    Args:
        model_path (str): Path to the trained model
        images_dir (str): Directory containing unlabelled images
        output_dir (str): Directory to save annotations
        conf_threshold (float): Confidence threshold for detections
        batch_size (int): Batch size for inference
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = YOLO(model_path)

    # Get a list of all images
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*.{ext}")))

    print(f"Found {len(image_files)} images in {images_dir}")

    # Process images in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[i:i+batch_size]

        # Run inference on the batch
        results = model(batch_files, conf=conf_threshold)

        # Process each result
        for result, image_path in zip(results, batch_files):
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]

            # Create the output file path
            output_file = os.path.join(output_dir, f"{base_name}.txt")

            # Open the output file for writing
            with open(output_file, 'w') as f:
                # Extract detections
                boxes = result.boxes
                for box in boxes:
                    # Get class ID and bounding box coordinates
                    cls_id = int(box.cls.item())
                    x, y, w, h = box.xywhn[0].tolist()  # Normalized xywh

                    # Write to the output file in YOLO format: class_id x_center y_center width height
                    f.write(f"{cls_id} {x} {y} {w} {h}\n")

    print(f"Annotations generated and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate annotations for unlabelled images.")
    parser.add_argument("--model", type=str, default="model/best.pt", help="Path to the trained model")
    parser.add_argument("--images", type=str, default="dataset_unlabelled", help="Directory containing unlabelled images")
    parser.add_argument("--output", type=str, default="generated_labels", help="Directory to save annotations")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    generate_annotations(
        model_path=args.model,
        images_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
        batch_size=args.batch_size
    )
