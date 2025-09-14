"""
Uncertainty Visualization
"""

# analyze.py

import os
import torch
from scene import Scene
from gaussian_renderer import render
import argparse
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# 导入我们之前创建的不确定性计算函数
from utils.uncertainty import compute_uncertainty_map

def visualize_and_save(output_path, base_name, rendered_image, uncertainty_map):
    """
    A helper function to visualize and save the output images.
    """
    # 1. Normalize uncertainty map for visualization
    normalized_entropy = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min())
    
    # 2. Apply a colormap to create a heatmap
    # Using 'inferno' as it shows high values (high uncertainty) as bright yellow
    heatmap_np = plt.get_cmap('inferno')(normalized_entropy.cpu().numpy())[..., :3]
    heatmap_tensor = torch.from_numpy(heatmap_np).permute(2, 0, 1).float() # HWC -> CHW

    # 3. Create a blended image for intuitive comparison
    alpha_blend = 0.6
    # Ensure heatmap is on the same device as the rendered image
    blended_image = rendered_image.cpu() * (1 - alpha_blend) + heatmap_tensor * alpha_blend
    
    # 4. Save the images
    vutils.save_image(rendered_image, os.path.join(output_path, f"{base_name}_render.png"))
    vutils.save_image(heatmap_tensor, os.path.join(output_path, f"{base_name}_heatmap.png"))
    vutils.save_image(blended_image, os.path.join(output_path, f"{base_name}_blended.png"))

def analyze_scene(args):
    """
    Main function to load a model and perform uncertainty analysis.
    """
    # Load the trained model
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Define which camera views to analyze
    if args.skip_train:
        # Only analyze the test/validation views
        viewpoints = scene.getTestCameras()
        print("Analyzing test cameras...")
    else:
        # Analyze all available cameras
        viewpoints = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
        print("Analyzing all cameras...")
        
    if not viewpoints:
        print("No cameras found to analyze. If using --skip_train, ensure your dataset has a test set.")
        return

    # Create output directory
    output_dir = os.path.join(args.model_path, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving analysis results to: {output_dir}")

    # Main analysis loop
    for viewpoint_cam in tqdm(viewpoints, desc="Analyzing Views"):
        # Render the scene from the current viewpoint
        # This calls the MODIFIED render function
        render_pkg = render(viewpoint_cam, gaussians, args, background)
        
        # Extract necessary data
        rendered_image = render_pkg["render"]
        rasterizer_out = render_pkg["rasterizer_out"]

        ############### DEBUG
        print("\nDEBUG: Type of rasterizer_out:", type(rasterizer_out))
        print("DEBUG: Length of rasterizer_out:", len(rasterizer_out))

        n_contrib = rasterizer_out['n_contrib']
        alphas = rasterizer_out['alphas']
        height, width = n_contrib.shape
        
        # Compute the uncertainty map
        uncertainty_map = compute_uncertainty_map(alphas, n_contrib, height, width)
        
        # Visualize and save the results
        image_name = os.path.basename(viewpoint_cam.image_name)
        visualize_and_save(output_dir, image_name, rendered_image, uncertainty_map)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Analysis script for trained PGSR models.")
    
    # Inherit arguments from the original PGSR training script
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    
    # Add custom arguments for this script
    parser.add_argument('--iteration', default=-1, type=int, help="Load model from a specific iteration, -1 for latest")
    parser.add_argument("--skip_train", action="store_true", help="If set, only renders test views")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    
    # Find the latest iteration if not specified
    if args.iteration == -1:
        # PGSR saves checkpoints in model_path/point_cloud/iteration_...
        point_cloud_dir = os.path.join(args.model_path, "point_cloud")
        try:
            available_iters = sorted([int(f.split("_")[1]) for f in os.listdir(point_cloud_dir) if f.startswith("iteration_")])
            args.iteration = available_iters[-1]
        except (FileNotFoundError, IndexError):
            print(f"Error: Could not find trained model checkpoints in {point_cloud_dir}")
            exit(1)

    print(f"Loading trained model from iteration {args.iteration}...")

    # Set up a safety hook for ctrl-c
    safe_state(args.quiet)

    # Run the analysis
    analyze_scene(args)