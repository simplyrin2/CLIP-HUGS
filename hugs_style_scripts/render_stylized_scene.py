#!/usr/bin/env python3
"""
Render Stylized Scene Script

This script renders the stylized scene checkpoint produced by style_scene_only.py
Uses the ML-HUGS evaluation system similar to evaluate.py

Usage Examples:

1. Render scene-only (no human model needed):
python scripts/render_stylized_scene.py \
  --scene_ckpt scene_only_stylized.pth \
  --mode scene \
  --output_dir renders/scene_only

2. Render combined with human model:
python scripts/render_stylized_scene.py \
  --scene_ckpt scene_only_stylized.pth \
  --human_ckpt /path/to/human_final.pth \
  --mode human_scene \
  --output_dir renders/combined

3. Render comparison (original vs stylized):
python scripts/render_stylized_scene.py \
  --scene_ckpt scene_only_stylized.pth \
  --original_scene_ckpt /path/to/original_scene_final.pth \
  --human_ckpt /path/to/human_final.pth \
  --mode comparison \
  --output_dir renders/comparison

4. Create rotating video:
python scripts/render_stylized_scene.py \
  --scene_ckpt scene_only_stylized.pth \
  --human_ckpt /path/to/human_final.pth \
  --mode video \
  --video_frames 120 \
  --output_dir renders/video
"""

import os
import sys
import json
import glob
import torch
import shutil
import argparse
import subprocess
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from runtime import configure_runtime

configure_runtime()

from hugs.trainer import GaussianTrainer
from hugs.utils.general import safe_state
from hugs.cfg.config import cfg as default_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Render Stylized Scene")
    
    # Model checkpoints
    parser.add_argument("--scene_ckpt", type=str, required=True,
                        help="Path to scene checkpoint (stylized or original)")
    parser.add_argument("--human_ckpt", type=str, default=None,
                        help="Path to human checkpoint")
    parser.add_argument("--original_scene_ckpt", type=str, default=None,
                        help="Path to original scene checkpoint (for comparison)")
    
    # Config and dataset
    parser.add_argument("--cfg_file", type=str, 
                        default=str(project_root / "configs" / "hugs_scene.yaml"),
                        help="Path to config file")
    parser.add_argument("--dataset_seq", type=str, default="bike",
                        help="Dataset sequence name")
    
    # Rendering mode
    parser.add_argument("--mode", type=str, default="scene",
                        choices=["scene", "human", "human_scene", "comparison", "video"],
                        help="Rendering mode")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/rendered_scene",
                        help="Output directory for renders")
    
    # Rendering parameters
    parser.add_argument("--render_frames", type=int, default=50,
                        help="Number of frames to render")
    parser.add_argument("--video_frames", type=int, default=120,
                        help="Number of frames for video mode")
    
    # Video parameters
    parser.add_argument("--camera_radius", type=float, default=2.5,
                        help="Camera distance from center")
    parser.add_argument("--camera_height", type=float, default=1.0,
                        help="Camera height")
    parser.add_argument("--create_video", action="store_true",
                        help="Create MP4 video from rendered frames")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS")
    
    # ML-HUGS specific options
    parser.add_argument("--eval", action="store_true", default=True,
                        help="Run in evaluation mode")
    parser.add_argument("--animate", action="store_true", default=True,
                        help="Generate animation")
    
    return parser.parse_args()


def get_logger(cfg):
    """Setup logger exactly like evaluate.py"""
    logdir = cfg.logdir
    mode = 'render_canon'
    
    os.makedirs(logdir, exist_ok=True)
    logger.add(os.path.join(cfg.logdir, f'{mode}.log'), level='INFO')
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))


def inspect_checkpoint(ckpt_path):
    """Inspect checkpoint to understand its structure"""
    if not os.path.exists(ckpt_path):
        return None
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check if it's a nested structure
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    logger.info(f"  {key} keys: {list(value.keys())}")
                else:
                    logger.info(f"  {key}: {type(value)}")
        
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint {ckpt_path}: {e}")
        return None


def setup_checkpoint_directory(cfg, args):
    """Setup checkpoint directory following evaluate.py pattern exactly"""
    
    # Create a temporary checkpoint directory like evaluate.py expects
    temp_ckpt_dir = os.path.join(args.output_dir, 'temp_ckpt')
    os.makedirs(temp_ckpt_dir, exist_ok=True)
    
    # Handle human checkpoint
    if args.human_ckpt and os.path.exists(args.human_ckpt):
        logger.info(f"Inspecting human checkpoint: {args.human_ckpt}")
        human_checkpoint = inspect_checkpoint(args.human_ckpt)
        
        if human_checkpoint is not None:
            # Copy human checkpoint with the naming pattern evaluate.py expects
            human_dest = os.path.join(temp_ckpt_dir, 'human_final.pth')
            shutil.copy2(args.human_ckpt, human_dest)
            logger.info(f'Copied human ckpt to: {human_dest}')
        else:
            logger.error(f'Failed to load human checkpoint: {args.human_ckpt}')
            return False
    elif args.mode in ['human', 'human_scene']:
        logger.error(f'Human ckpt is required for {args.mode} mode.')
        return False
    else:
        logger.warning('No human ckpt provided')
    
    # Handle scene checkpoint
    if args.scene_ckpt and os.path.exists(args.scene_ckpt):
        logger.info(f"Inspecting scene checkpoint: {args.scene_ckpt}")
        scene_checkpoint = inspect_checkpoint(args.scene_ckpt)
        
        if scene_checkpoint is not None:
            # Copy scene checkpoint with the naming pattern evaluate.py expects
            scene_dest = os.path.join(temp_ckpt_dir, 'scene_final.pth')
            shutil.copy2(args.scene_ckpt, scene_dest)
            logger.info(f'Copied scene ckpt to: {scene_dest}')
        else:
            logger.error(f'Failed to load scene checkpoint: {args.scene_ckpt}')
            return False
    elif args.mode in ['scene', 'human_scene']:
        logger.error(f'Scene ckpt is required for {args.mode} mode.')
        return False
    else:
        logger.warning('No scene ckpt provided')
    
    # Update cfg to point to temp directory (like evaluate.py does)
    cfg.logdir_ckpt = temp_ckpt_dir
    
    return True


def create_video_from_frames(frame_dir, output_path, fps=30, frame_pattern="*.png"):
    """Create MP4 video from rendered frames"""
    import subprocess
    
    # Find all PNG files in directory
    frame_files = sorted(glob.glob(os.path.join(frame_dir, frame_pattern)))
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return False
    
    # Create a temporary file list for ffmpeg
    temp_list = os.path.join(frame_dir, "frame_list.txt")
    with open(temp_list, 'w') as f:
        for frame_file in frame_files:
            f.write(f"file '{frame_file}'\n")
    
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', temp_list,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-r', str(fps), '-crf', '18',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_list)  # Clean up
        print(f"Video created: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        if os.path.exists(temp_list):
            os.remove(temp_list)
        return False


def render_stylized_scene(cfg, args):
    """Render stylized scene using ML-HUGS trainer - exactly like evaluate.py"""
    
    # Setup checkpoint directory exactly like evaluate.py
    if not setup_checkpoint_directory(cfg, args):
        return
    
    # Now use evaluate.py's exact pattern for finding checkpoints
    latest_human_ckpt = None
    human_ckpt_files = glob.glob(cfg.logdir_ckpt + '/*human*.pth')
    human_ckpt_files += glob.glob(cfg.logdir_ckpt + '/ckpt/*human*.pth')
    human_ckpt_files = sorted(human_ckpt_files)
    
    if len(human_ckpt_files) > 0:
        latest_human_ckpt = human_ckpt_files[-1]
        logger.info(f'Found human ckpt: {latest_human_ckpt}')
        cfg.human.ckpt = latest_human_ckpt
    else:
        if cfg.mode in ['human', 'human_scene']:
            logger.error(f'Human ckpt is required for {cfg.mode} mode.')
            return
        else:
            logger.warning('No human ckpt found')
            
    latest_scene_ckpt = None
    scene_ckpt_files = sorted(glob.glob(cfg.logdir_ckpt + '/*scene*.pth'))
    if len(scene_ckpt_files) > 0:
        latest_scene_ckpt = scene_ckpt_files[-1]
        logger.info(f'Found scene ckpt: {latest_scene_ckpt}')
        cfg.scene.ckpt = latest_scene_ckpt
    else:
        if cfg.mode in ['scene', 'human_scene']:
            logger.error(f'Scene ckpt is required for {cfg.mode} mode.')
            return
        else:
            logger.warning('No scene ckpt found')
    
    # Create trainer exactly like evaluate.py
    trainer = GaussianTrainer(cfg)
    
    # Run validation
    logger.info('Running validation...')
    trainer.validate()
                
    # Run animation if requested
    if args.animate:
        logger.info('Generating animation...')
        trainer.animate()
    
    # Save evaluation metrics exactly like evaluate.py
    mode = 'eval' if cfg.eval else 'train'
    with open(os.path.join(cfg.logdir, f'results_{mode}.json'), 'w') as f:
        json.dump(trainer.eval_metrics, f, indent=4)


def render_comparison(cfg, args):
    """Render comparison between original and stylized scene"""
    
    if not args.original_scene_ckpt or not args.scene_ckpt:
        logger.error('Both original_scene_ckpt and scene_ckpt required for comparison')
        return
    
    # Create subdirectories
    original_dir = os.path.join(args.output_dir, 'original')
    stylized_dir = os.path.join(args.output_dir, 'stylized')
    comparison_dir = os.path.join(args.output_dir, 'comparison')
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(stylized_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    logger.info('Rendering original scene...')
    
    # Render original scene
    cfg_orig = OmegaConf.create(cfg)
    cfg_orig.logdir = original_dir
    args_orig = argparse.Namespace(**vars(args))
    args_orig.scene_ckpt = args.original_scene_ckpt
    args_orig.output_dir = original_dir
    
    render_stylized_scene(cfg_orig, args_orig)
    
    logger.info('Rendering stylized scene...')
    
    # Render stylized scene
    cfg_style = OmegaConf.create(cfg)
    cfg_style.logdir = stylized_dir
    args_style = argparse.Namespace(**vars(args))
    args_style.output_dir = stylized_dir
    
    render_stylized_scene(cfg_style, args_style)
    
    logger.info('Creating side-by-side comparisons...')
    
    # Create side-by-side comparisons
    original_frames = sorted(glob.glob(os.path.join(original_dir, 'val', '*.png')))
    stylized_frames = sorted(glob.glob(os.path.join(stylized_dir, 'val', '*.png')))
    
    for i, (orig_path, style_path) in enumerate(zip(original_frames, stylized_frames)):
        if os.path.exists(orig_path) and os.path.exists(style_path):
            # Load images
            orig_img = cv2.imread(orig_path)
            style_img = cv2.imread(style_path)
            
            if orig_img is not None and style_img is not None:
                # Resize if needed
                if orig_img.shape != style_img.shape:
                    style_img = cv2.resize(style_img, (orig_img.shape[1], orig_img.shape[0]))
                
                # Create side-by-side comparison
                comparison = np.hstack([orig_img, style_img])
                
                # Save comparison
                comparison_path = os.path.join(comparison_dir, f'comparison_{i:04d}.png')
                cv2.imwrite(comparison_path, comparison)
    
    logger.info(f'Comparison frames saved to: {comparison_dir}')
    
    # Create comparison video if requested
    if args.create_video:
        video_path = os.path.join(args.output_dir, 'comparison_video.mp4')
        create_video_from_frames(comparison_dir, video_path, args.fps)


def render_video_sequence(cfg, args):
    """Render rotating video sequence"""
    
    logger.info('Rendering rotating video sequence...')
    
    # Setup checkpoints
    if not setup_checkpoint_directory(cfg, args):
        return
    
    # Use evaluate.py pattern for checkpoint discovery
    cfg.mode = 'human_scene' if args.human_ckpt else 'scene'
    
    # Configure animation parameters
    if hasattr(cfg, 'animation'):
        cfg.animation.num_frames = args.video_frames
        cfg.animation.camera_radius = args.camera_radius
        cfg.animation.camera_height = args.camera_height
    
    # Run the same logic as render_stylized_scene
    render_stylized_scene(cfg, args)
    
    # Create video from animation frames
    if args.create_video:
        anim_dir = os.path.join(args.output_dir, 'anim')
        if os.path.exists(anim_dir):
            video_path = os.path.join(args.output_dir, 'stylized_video.mp4')
            create_video_from_frames(anim_dir, video_path, args.fps)
        else:
            logger.warning(f'Animation directory not found: {anim_dir}')


@torch.no_grad()
def main():
    args = parse_args()
    
    # Setup output directory FIRST
    os.makedirs(args.output_dir, exist_ok=True)
    safe_state(seed=42)  # Set random seed for reproducibility
    
    print("=== Stylized Scene Rendering ===")
    print(f"Scene checkpoint: {args.scene_ckpt}")
    print(f"Human checkpoint: {args.human_ckpt}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    
    # Load configuration exactly like evaluate.py
    if os.path.exists(args.cfg_file):
        cfg_file = OmegaConf.load(args.cfg_file)
    else:
        print(f'Config file not found: {args.cfg_file}')
        return
    
    # Merge with default config exactly like evaluate.py
    cfg = OmegaConf.merge(default_cfg, cfg_file)
    cfg.eval = True  # Always in eval mode
    
    # Set dataset sequence
    cfg.dataset.seq = args.dataset_seq
    
    # Set mode
    cfg.mode = args.mode
    
    # Set directories exactly like evaluate.py
    cfg.logdir = args.output_dir
    cfg.logdir_ckpt = args.output_dir  # Will be updated by setup_checkpoint_directory
    
    # Setup logger exactly like evaluate.py
    get_logger(cfg)
    
    try:
        if args.mode == "comparison":
            render_comparison(cfg, args)
        elif args.mode == "video":
            render_video_sequence(cfg, args)
        else:
            render_stylized_scene(cfg, args)
        
        print(f"\nRendering complete! Output saved to: {args.output_dir}")
        
        # List output files
        val_dir = os.path.join(args.output_dir, 'val')
        anim_dir = os.path.join(args.output_dir, 'anim')
        
        if os.path.exists(val_dir):
            val_files = glob.glob(os.path.join(val_dir, '*.png'))
            print(f"Validation frames: {len(val_files)} files in {val_dir}")
        
        if os.path.exists(anim_dir):
            anim_files = glob.glob(os.path.join(anim_dir, '*.png'))
            print(f"Animation frames: {len(anim_files)} files in {anim_dir}")
        
        # Clean up temporary checkpoint directory
        temp_ckpt_dir = os.path.join(args.output_dir, 'temp_ckpt')
        if os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir)
            print("Cleaned up temporary checkpoint directory")
        
    except Exception as e:
        logger.error(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()