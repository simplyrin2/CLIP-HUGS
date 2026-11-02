#!/usr/bin/env python3
"""
Scene Style Transfer Script

This script applies style transfer to scene components while keeping human models frozen.
It uses the ML-HUGS style system with CLIP losses to modify only the background scene.

Usage:
python scripts/style_scene.py \
  --cfg_file cfg_files/release/neuman/hugs_scene.yaml \
  --dataset_seq bike \
  --human_ckpt /path/to/human_final.pth \
  --scene_ckpt /path/to/scene_final.pth \
  --style_text "a Van Gogh painting with swirling brushstrokes" \
  --steps 5000 \
  --out_ckpt scene_stylized.pth

Or with style image reference:
python scripts/style_scene.py \
  --cfg_file cfg_files/release/neuman/hugs_scene.yaml \
  --dataset_seq bike \
  --human_ckpt /path/to/human_final.pth \
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

from hugs_style_module.scene_adapter import SceneAdapter
from hugs_style_module.scene_style_trainer import run_scene_style, run_scene_style_with_reference


def parse_args():
    parser = argparse.ArgumentParser(description="Scene Style Transfer")
    
    # Model and data paths
    parser.add_argument("--cfg_file", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--dataset_seq", type=str, required=True,
                        help="Dataset sequence name")
    parser.add_argument("--human_ckpt", type=str, default=None,
                        help="Path to human model checkpoint (optional for scene-only mode)")
    parser.add_argument("--scene_only", action="store_true",
                        help="Scene-only mode (no human model required)")
    parser.add_argument("--scene_ckpt", type=str, required=True,
                        help="Path to scene model checkpoint")
    
    # Style parameters
    parser.add_argument("--style_text", type=str, default=None,
                        help="Style text description")
    parser.add_argument("--style_image", type=str, default=None,
                        help="Path to style reference image")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate")
    
    # Loss weights
    parser.add_argument("--lambda_d", type=float, default=600.0,
                        help="Directional CLIP loss weight")
    parser.add_argument("--lambda_p", type=float, default=300.0,
                        help="Patch CLIP loss weight")
    parser.add_argument("--lambda_c", type=float, default=0.1,
                        help="Content preservation weight")
    parser.add_argument("--lambda_b", type=float, default=2.0,
                        help="Background uniformity weight")
    parser.add_argument("--lambda_scene", type=float, default=50.0,
                        help="Scene-only stylization weight")
    
    # Output
    parser.add_argument("--out_ckpt", type=str, default="scene_stylized.pth",
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
    
    # Validate scene-only mode
    if args.scene_only and args.human_ckpt is not None:
        print("Warning: --scene_only specified but --human_ckpt provided. Using scene-only mode.")
        args.human_ckpt = None
    
    if not args.scene_only and args.human_ckpt is None:
        raise ValueError("Must provide --human_ckpt unless using --scene_only mode")
    
    # Check if checkpoints exist
    if args.human_ckpt and not os.path.exists(args.human_ckpt):
        raise FileNotFoundError(f"Human checkpoint not found: {args.human_ckpt}")
    
    if not os.path.exists(args.scene_ckpt):
        raise FileNotFoundError(f"Scene checkpoint not found: {args.scene_ckpt}")
    
    print("=== Scene Style Transfer ===")
    print(f"Human checkpoint: {args.human_ckpt}")
    print(f"Scene checkpoint: {args.scene_ckpt}")
    print(f"Dataset sequence: {args.dataset_seq}")
    print(f"Output checkpoint: {args.out_ckpt}")
    
    if args.style_text:
        print(f"Style text: '{args.style_text}'")
    else:
        print(f"Style image: {args.style_image}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize scene adapter
    print("\nInitializing scene adapter...")
    adapter = SceneAdapter(
        cfg_file=args.cfg_file,
        dataset_seq=args.dataset_seq,
        human_checkpoint=args.human_ckpt,
        scene_checkpoint=args.scene_ckpt,
        device=device,
        scene_only=args.scene_only
    )
    
    print(f"Loaded {adapter.num_frames} frames")
    print(f"Trainable scene parameters: {len(adapter.scene_parameters())}")
    
    # Run style transfer
    print(f"\nStarting style transfer training...")
    
    if args.style_text:
        # Text-based style transfer
        final_checkpoint = run_scene_style(
            adapter=adapter,
            style_text=args.style_text,
            steps=args.steps,
            lr=args.lr,
            lambda_d=args.lambda_d,
            lambda_p=args.lambda_p,
            lambda_c=args.lambda_c,
            lambda_b=args.lambda_b,
            lambda_scene=args.lambda_scene,
            out_ckpt=args.out_ckpt,
            device=args.device,
            save_interval=args.save_interval
        )
    else:
        # Image-based style transfer
        final_checkpoint = run_scene_style_with_reference(
            adapter=adapter,
            style_image_path=args.style_image,
            steps=args.steps,
            lr=args.lr,
            lambda_d=args.lambda_d,
            lambda_p=args.lambda_p,
            lambda_c=args.lambda_c,
            lambda_b=args.lambda_b,
            lambda_scene=args.lambda_scene,
            out_ckpt=args.out_ckpt,
            device=args.device
        )
    
    print(f"\n=== Style Transfer Complete ===")
    print(f"Stylized scene saved to: {final_checkpoint}")
    print(f"You can now use this checkpoint to render stylized scenes!")
    
    # Provide usage instructions
    print(f"\nNext steps:")
    print(f"1. Use the stylized checkpoint for rendering:")
    print(f"   python scripts/evaluate.py --scene_ckpt {final_checkpoint} --human_ckpt {args.human_ckpt}")
    print(f"2. Or use with the rendering script:")
    print(f"   python ml_hugs_style_render.py --stylized_scene_checkpoint {final_checkpoint}")


if __name__ == "__main__":
    main()