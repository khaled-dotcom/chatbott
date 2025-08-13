import os
import shutil
from ultralytics import YOLO
import torch

def segment_image(input_path, output_dir, model_path, conf=0.3):
    """
    Segment an image using the YOLO model and save the result.
    
    Args:
        input_path (str): Path to the input image.
        output_dir (str): Directory to save the segmented image.
        model_path (str): Path to the YOLO model file.
        conf (float): Confidence threshold for predictions.
    
    Returns:
        str: Filename of the segmented image or None if an error occurs.
    """
    try:
        # Load YOLO model with weights_only=True
        model = YOLO(model_path)
        
        # Run segmentation
        results = model.predict(source=input_path, conf=conf, save=True)
        
        # Get the segmented image path
        run_dir = results[0].save_dir
        filename = os.path.basename(input_path)
        segmented_image = os.path.join(run_dir, filename)
        
        # Move segmented image to output directory
        output_filename = 'segmented_image.jpg'
        output_filepath = os.path.join(output_dir, output_filename)
        shutil.move(segmented_image, output_filepath)
        
        # Clean up YOLO runs directory
        shutil.rmtree(run_dir)
        
        return output_filename
    except Exception as e:
        print(f"Error in segmentation: {str(e)}")
        return None