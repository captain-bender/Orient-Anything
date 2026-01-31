"""
Inference script to run the Orient-Anything model on all images in dataset/benchmark folder
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

from paths import DINO_LARGE
from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
from inference import get_3angle, get_3angle_infer_aug
from utils import background_preprocess, render_3D_axis, overlay_images_with_scaling

# Configuration
BENCHMARK_FOLDER = "dataset/benchmark"
OUTPUT_FOLDER = "benchmark_results"
VISUALIZATIONS_FOLDER = "visualizations"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_INFERENCE_AUG = False  # Set to True to use inference augmentation
REMOVE_BACKGROUND = True  # Set to True to remove background

def load_model(device):
    """Load the Orient-Anything model"""
    print(f"Loading model on device: {device}")
    
    # Initialize model
    dino = DINOv2_MLP(
        dino_mode='large',
        in_dim=1024,
        out_dim=360+180+360+2,
        evaluate=True,
        mask_dino=False,
        frozen_back=False
    )
    dino.eval()
    
    # Load weights
    try:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="Viglong/Orient-Anything",
            filename="ronormsigma1/dino_weight.pt",
            repo_type="model",
            cache_dir='./',
            resume_download=True
        )
        print(f"Model checkpoint loaded from: {ckpt_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Trying to load from local cache...")
        ckpt_path = "models--Viglong--Orient-Anything/snapshots/ecfa089828d9fab521d47133bedbdbfdef151cd4/ronormsigma1/dino_weight.pt"
    
    dino.load_state_dict(torch.load(ckpt_path, map_location=device))
    dino = dino.to(device)
    print("Model weights loaded successfully")
    
    # Load preprocessor
    val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')
    
    return dino, val_preprocess

def run_inference_on_image(image_path, dino, val_preprocess, device):
    """Run inference on a single image and return results with timing"""
    start_time = time.time()
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        if USE_INFERENCE_AUG:
            origin_img = image
            rm_bkg_img = background_preprocess(origin_img, True)
            angles = get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, device)
        else:
            rm_bkg_img = background_preprocess(image, REMOVE_BACKGROUND)
            angles = get_3angle(rm_bkg_img, dino, val_preprocess, device)
        
        # Extract angles
        heading = float(angles[0])
        pitch = float(angles[1])
        roll = float(angles[2])
        confidence = float(angles[3])
        
        # Generate visualization if confidence is high enough
        visualization = None
        if confidence > 0.5:
            phi = np.radians(heading)
            theta = np.radians(pitch)
            gamma = roll
            render_axis = render_3D_axis(phi, theta, gamma)
            visualization = overlay_images_with_scaling(render_axis, rm_bkg_img)
        
        elapsed_time = time.time() - start_time
        
        return {
            "status": "success",
            "azimuth_deg": heading,
            "polar_deg": pitch,
            "rotation_deg": roll,
            "confidence_score": confidence,
            "visualization": visualization,
            "inference_time_seconds": elapsed_time
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "status": "error",
            "error_message": str(e),
            "visualization": None,
            "inference_time_seconds": elapsed_time
        }

def main():
    """Main inference script"""
    
    # Create output directories
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    results_visualizations_folder = os.path.join(OUTPUT_FOLDER, VISUALIZATIONS_FOLDER)
    os.makedirs(results_visualizations_folder, exist_ok=True)
    
    # Load model
    dino, val_preprocess = load_model(DEVICE)
    
    # Get all images from benchmark folder
    benchmark_path = Path(BENCHMARK_FOLDER)
    image_files = sorted(list(benchmark_path.glob("*.png")) + list(benchmark_path.glob("*.jpg")) + list(benchmark_path.glob("*.jpeg")))
    
    if not image_files:
        print(f"No images found in {BENCHMARK_FOLDER}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Using inference augmentation: {USE_INFERENCE_AUG}")
    print(f"Removing background: {REMOVE_BACKGROUND}")
    print(f"Device: {DEVICE}\n")
    
    # Run inference
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_folder": BENCHMARK_FOLDER,
            "total_images": len(image_files),
            "inference_augmentation": USE_INFERENCE_AUG,
            "background_removal": REMOVE_BACKGROUND,
            "device": DEVICE
        },
        "predictions": []
    }
    
    successful = 0
    failed = 0
    total_inference_time = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        image_name = image_path.name
        image_stem = image_path.stem
        
        print(f"\nProcessing: {image_name}")
        result = run_inference_on_image(str(image_path), dino, val_preprocess, DEVICE)
        inference_time = result.get("inference_time_seconds", 0)
        total_inference_time += inference_time
        
        if result["status"] == "success":
            successful += 1
            print(f"  ✓ Azimuth: {result['azimuth_deg']:.2f}°, Polar: {result['polar_deg']:.2f}°, Rotation: {result['rotation_deg']:.2f}°, Confidence: {result['confidence_score']:.4f}")
            print(f"  ⏱ Inference time: {inference_time:.3f}s")
            
            # Save visualization if available
            if result["visualization"] is not None:
                viz_path = os.path.join(results_visualizations_folder, f"{image_stem}_axis.png")
                result["visualization"].save(viz_path)
                print(f"  → Visualization saved: {viz_path}")
                result["visualization_file"] = f"{VISUALIZATIONS_FOLDER}/{image_stem}_axis.png"
            else:
                print(f"  → No visualization (confidence too low: {result['confidence_score']:.4f})")
                result["visualization_file"] = None
        else:
            failed += 1
            print(f"  ✗ Error: {result.get('error_message', 'Unknown error')}")
            print(f"  ⏱ Inference time: {inference_time:.3f}s")
            result["visualization_file"] = None
        
        # Remove PIL Image object before saving to JSON
        result.pop("visualization", None)
        
        # Add to results
        result["image_file"] = image_name
        results["predictions"].append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"{'='*60}")
    print(f"Total processed: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if successful > 0:
        print(f"Total inference time: {total_inference_time:.3f}s")
        print(f"Average inference time: {total_inference_time/successful:.3f}s per image")
    print(f"{'='*60}\n")
    
    # Save results
    output_file = os.path.join(OUTPUT_FOLDER, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"Visualizations saved to: {results_visualizations_folder}")
    
    return results

if __name__ == "__main__":
    main()
