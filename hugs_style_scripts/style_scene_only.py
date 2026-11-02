#!/usr/bin/env python3
"""
Scene-Only Style Transfer Script

This script applies style transfer to ONLY the scene components without needing human models.
This is more efficient when you only want to stylize backgrounds.

Usage:
python scripts/style_scene_only.py \
  --cfg_file cfg_files/release/neuman/hugs_scene.yaml \
  --dataset_seq bike \
  --scene_ckpt /path/to/scene_final.pth \
  --style_text "a Van Gogh painting with swirling brushstrokes" \
  --steps 5000 \
  --out_ckpt scene_stylized.pth

Or with style image reference:
python scripts/style_scene_only.py \
  --cfg_file cfg_files/release/neuman/hugs_scene.yaml \
  --dataset_seq bike \
  --scene_ckpt /path/to/scene_final.pth \
  --style_image /path/to/style_reference.jpg \
  --steps 3000 \
  --out_ckpt scene_stylized.pth
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from runtime import configure_runtime

configure_runtime()

import torch

from hugs_style_module.scene_style_trainer import (
    run_gaussian_editor_style_transfer,
    run_gaussian_editor_style_with_image,
    convert_mlhugs_to_gaussian_editor,
    save_gaussian_editor_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scene-Only Style Transfer")
    
    # Model and data paths
    parser.add_argument("--cfg_file", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--dataset_seq", type=str, required=True,
                        help="Dataset sequence name")
    parser.add_argument("--scene_ckpt", type=str, required=True,
                        help="Path to scene model checkpoint")
    
    # Style parameters
    parser.add_argument("--style_text", type=str, default=None,
                        help="Style text description")
    parser.add_argument("--style_image", type=str, default=None,
                        help="Path to style reference image")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=3000,
                        help="Number of training steps")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    
    # Loss weights (adjusted for scene-only)
    parser.add_argument("--lambda_d", type=float, default=20.0,
                        help="Directional CLIP loss weight")
    parser.add_argument("--lambda_p", type=float, default=200.0,
                        help="Patch CLIP loss weight")
    parser.add_argument("--lambda_c", type=float, default=0.5,
                        help="Content preservation weight (lower for scene-only)")
    parser.add_argument("--lambda_b", type=float, default=10.0,
                        help="Background uniformity weight")
    parser.add_argument("--lambda_scene", type=float, default=100.0,
                        help="Scene-only stylization weight (higher for scene-only)")
    
    # Output
    parser.add_argument("--out_ckpt", type=str, default="scene_only_stylized.pth",
                        help="Output checkpoint path")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Steps between checkpoint saves")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate arguments
    if args.style_text is None and args.style_image is None:
        raise ValueError("Must provide either --style_text or --style_image")
    
    if args.style_text is not None and args.style_image is not None:
        raise ValueError("Provide either --style_text OR --style_image, not both")
    
    # Check if checkpoint exists
    if not os.path.exists(args.scene_ckpt):
        raise FileNotFoundError(f"Scene checkpoint not found: {args.scene_ckpt}")
    
    print("=== Scene-Only Style Transfer ===")
    print(f"Scene checkpoint: {args.scene_ckpt}")
    print(f"Dataset sequence: {args.dataset_seq}")
    print(f"Output checkpoint: {args.out_ckpt}")
    print("Human model: NOT REQUIRED (scene-only mode)")
    
    if args.style_text:
        print(f"Style text: '{args.style_text}'")
    else:
        print(f"Style image: {args.style_image}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run style transfer using GaussianEditor's GaussianModel
    print(f"\nStarting GaussianEditor-based scene style transfer...")
    
    if args.style_text:
        # Text-based style transfer
        print(f"Using text prompt: '{args.style_text}'")
        final_checkpoint = run_gaussian_editor_style_transfer(
            scene_checkpoint_path=args.scene_ckpt,
            style_text=args.style_text,
            steps=args.steps,
            lr=args.lr,
            lambda_d=args.lambda_d,
            lambda_p=args.lambda_p,
            lambda_c=args.lambda_c,
            lambda_b=args.lambda_b,
            out_ckpt=args.out_ckpt,
            device=args.device,
            save_interval=args.save_interval
        )
    else:
        # Image-based style transfer
        print(f"Using style image: {args.style_image}")
        final_checkpoint = run_gaussian_editor_style_with_image(
            scene_checkpoint_path=args.scene_ckpt,
            style_image_path=args.style_image,
            steps=args.steps,
            lr=args.lr,
            out_ckpt=args.out_ckpt,
            device=args.device
        )
    
    print(f"\n=== Scene-Only Style Transfer Complete ===")
    print(f"Stylized scene saved to: {final_checkpoint}")
    
    print(f"\nTo use this stylized scene:")
    print(f"1. With original human model for combined rendering:")
    print(f"   python scripts/evaluate.py --scene_ckpt {final_checkpoint} --human_ckpt /path/to/human_final.pth")
    print(f"2. Scene-only rendering:")
    print(f"   python scripts/evaluate.py --scene_ckpt {final_checkpoint} --mode scene")
    print(f"3. The scene checkpoint can be used with any human model from the same dataset!")


if __name__ == "__main__":
    main()