#!/usr/bin/env python3
"""
Evaluate Stylized Scene Script

This script renders both ground truth and stylized scenes, then computes comprehensive
evaluation metrics including CLIP scores, PSNR, and LPIPS.

Usage Examples:

1. Evaluate scene-only stylization:
python scripts/evaluate_stylized_scene.py \
  --scene_ckpt /path/to/scene_stylized.pth \
  --original_scene_ckpt /path/to/scene_final.pth \
  --mode scene \
  --output_dir evaluation/scene_only

2. Evaluate with human model:
python scripts/evaluate_stylized_scene.py \
  --scene_ckpt scene_stylized.pth \
  --original_scene_ckpt scene_original.pth \
  --human_ckpt human_final.pth \
  --mode human_scene \
  --output_dir evaluation/combined

3. Style transfer evaluation:
python scripts/evaluate_stylized_scene.py \
  --scene_ckpt scene_stylized.pth \
  --original_scene_ckpt scene_original.pth \
  --style_text "Van Gogh painting" \
  --mode comparison \
  --output_dir evaluation/style_transfer
"""

import os
import sys
import json
import glob
import torch
import torch.nn.functional as F
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
import lpips
import clip

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from runtime import configure_runtime

configure_runtime()

from hugs.trainer import GaussianTrainer
from hugs.utils.general import safe_state
from hugs.cfg.config import cfg as default_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stylized Scene")
    
    # Model checkpoints
    parser.add_argument("--scene_ckpt", type=str, required=True,
                        help="Path to stylized scene checkpoint")
    parser.add_argument("--original_scene_ckpt", type=str, required=True,
                        help="Path to original scene checkpoint (ground truth)")
    parser.add_argument("--human_ckpt", type=str, default=None,
                        help="Path to human checkpoint")
    
    # Config and dataset
    parser.add_argument("--cfg_file", type=str, 
                        default=str(project_root / "configs" / "hugs_scene.yaml"),
                        help="Path to config file")
    parser.add_argument("--dataset_seq", type=str, default="bike",
                        help="Dataset sequence name")
    
    # Style evaluation
    parser.add_argument("--style_text", type=str, default="",
                        help="Style text prompt for directional CLIP evaluation")
    parser.add_argument("--style_image", type=str, default="",
                        help="Style reference image path for CLIP evaluation")
    
    # Rendering mode
    parser.add_argument("--mode", type=str, default="scene",
                        choices=["scene", "human", "human_scene", "comparison"],
                        help="Evaluation mode")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    
    # Evaluation parameters
    parser.add_argument("--num_views", type=int, default=50,
                        help="Number of views to evaluate")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image resolution for evaluation")
    parser.add_argument("--force_split_sidebyside", action="store_true",
                        help="Force splitting of side-by-side comparison images")
    
    # ML-HUGS specific options
    parser.add_argument("--eval", action="store_true", default=True,
                        help="Run in evaluation mode")
    
    return parser.parse_args()


def get_logger(cfg):
    """Setup logger"""
    logdir = cfg.logdir
    mode = 'evaluate_stylized'
    
    os.makedirs(logdir, exist_ok=True)
    logger.add(os.path.join(cfg.logdir, f'{mode}.log'), level='INFO')
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))


def setup_checkpoint_directory(cfg, args, ckpt_path, ckpt_name):
    """Setup checkpoint directory for a specific checkpoint"""
    temp_ckpt_dir = os.path.join(args.output_dir, f'temp_ckpt_{ckpt_name}')
    os.makedirs(temp_ckpt_dir, exist_ok=True)
    
    # Handle human checkpoint
    if args.human_ckpt and os.path.exists(args.human_ckpt):
        human_dest = os.path.join(temp_ckpt_dir, 'human_final.pth')
        shutil.copy2(args.human_ckpt, human_dest)
        logger.info(f'Copied human ckpt to: {human_dest}')
    elif args.mode in ['human', 'human_scene']:
        logger.error(f'Human ckpt is required for {args.mode} mode.')
        return None
    
    # Handle scene checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        scene_dest = os.path.join(temp_ckpt_dir, 'scene_final.pth')
        shutil.copy2(ckpt_path, scene_dest)
        logger.info(f'Copied scene ckpt to: {scene_dest}')
    else:
        logger.error(f'Scene ckpt not found: {ckpt_path}')
        return None
    
    return temp_ckpt_dir


def render_scene_with_checkpoint(cfg, args, checkpoint_path, output_subdir):
    """Render scenes using a specific checkpoint"""
    
    # Setup checkpoint directory
    temp_ckpt_dir = setup_checkpoint_directory(cfg, args, checkpoint_path, output_subdir)
    if not temp_ckpt_dir:
        return None
    
    # Create output directory for this rendering
    render_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(render_dir, exist_ok=True)
    
    # Configure for this specific rendering
    cfg_render = OmegaConf.create(cfg)
    cfg_render.logdir = render_dir
    cfg_render.logdir_ckpt = temp_ckpt_dir
    cfg_render.mode = args.mode
    # Disable side-by-side rendering to get clean evaluation images
    cfg_render.train.save_side_by_side = False
    
    # Find checkpoints in temp directory
    human_ckpt_files = glob.glob(temp_ckpt_dir + '/*human*.pth')
    scene_ckpt_files = glob.glob(temp_ckpt_dir + '/*scene*.pth')
    
    if len(scene_ckpt_files) > 0:
        cfg_render.scene.ckpt = scene_ckpt_files[0]
        logger.info(f'Using scene ckpt: {scene_ckpt_files[0]}')
    
    if len(human_ckpt_files) > 0:
        cfg_render.human.ckpt = human_ckpt_files[0]
        logger.info(f'Using human ckpt: {human_ckpt_files[0]}')
    elif args.mode in ['human', 'human_scene']:
        logger.error(f'Human ckpt is required for {args.mode} mode.')
        return None
    
    # Create trainer and run validation
    trainer = GaussianTrainer(cfg_render)
    logger.info(f'Running validation for {output_subdir}...')
    trainer.validate()
    
    # Clean up temp directory
    shutil.rmtree(temp_ckpt_dir)
    
    # Return path to rendered images
    val_dir = os.path.join(render_dir, 'val')
    return val_dir if os.path.exists(val_dir) else None


def load_clip_model(device):
    """Load CLIP model for evaluation"""
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return clip_model, preprocess


def load_lpips_model(device):
    """Load LPIPS model for perceptual evaluation"""
    lpips_model = lpips.LPIPS(net='alex').to(device)
    return lpips_model


def compute_psnr(img1, img2):
    """Compute PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_lpips(img1, img2, lpips_model, device):
    """Compute LPIPS perceptual distance"""
    # Convert to tensors and normalize to [-1, 1]
    img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float().to(device) / 255.0 * 2 - 1
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).float().to(device) / 255.0 * 2 - 1
    
    # Add batch dimension
    img1_tensor = img1_tensor.unsqueeze(0)
    img2_tensor = img2_tensor.unsqueeze(0)
    
    with torch.no_grad():
        lpips_score = lpips_model(img1_tensor, img2_tensor).item()
    
    return lpips_score


def compute_clip_score(image, text, clip_model, preprocess, device):
    """Compute CLIP score between image and text"""
    # Preprocess image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8))
    else:
        image_pil = image
    
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    text_tensor = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_features = clip_model.encode_text(text_tensor)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score


def compute_clip_directional_similarity(original_img, stylized_img, style_text, clip_model, preprocess, device):
    """Compute CLIP directional similarity for style transfer evaluation"""
    
    # Get CLIP features for all inputs
    def get_clip_features(img):
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img.astype(np.uint8))
        else:
            img_pil = img
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    # Get text features
    text_tensor = clip.tokenize([style_text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tensor)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Get image features
    original_features = get_clip_features(original_img)
    stylized_features = get_clip_features(stylized_img)
    
    # Compute directional similarity
    # Direction in CLIP space from original to stylized
    img_direction = stylized_features - original_features
    img_direction = img_direction / img_direction.norm(dim=-1, keepdim=True)
    
    # Direction from original image to style text
    text_direction = text_features - original_features
    text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity between directions
    directional_similarity = torch.matmul(img_direction, text_direction.T).item()
    
    return directional_similarity


def compute_clip_consistency(original_img, stylized_img, clip_model, preprocess, device):
    """Compute CLIP consistency (content preservation)"""
    
    def get_clip_features(img):
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img.astype(np.uint8))
        else:
            img_pil = img
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    original_features = get_clip_features(original_img)
    stylized_features = get_clip_features(stylized_img)
    
    # Compute cosine similarity
    consistency = torch.matmul(original_features, stylized_features.T).item()
    
    return consistency


def split_side_by_side_image(img):
    """Split a side-by-side image into left and right parts"""
    height, width = img.shape[:2]
    mid_width = width // 2
    
    # Left half (ground truth), right half (rendered)
    left_img = img[:, :mid_width]
    right_img = img[:, mid_width:]
    
    return left_img, right_img


def is_side_by_side_image(img):
    """Detect if image is likely a side-by-side comparison"""
    height, width = img.shape[:2]
    
    # Check if width is roughly twice the height (common for side-by-side)
    aspect_ratio = width / height
    
    # Also check if the image can be split evenly
    if width % 2 == 0 and aspect_ratio > 3.0:  # Very wide images are likely side-by-side
        return True
    
    # Additional check: compare left and right halves for similarity
    # If they're very different, it's likely side-by-side
    if width % 2 == 0 and width > height * 2:
        mid_width = width // 2
        left_half = img[:, :mid_width]
        right_half = img[:, mid_width:]
        
        # Simple difference check
        diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        if diff > 30:  # Significant difference suggests side-by-side
            return True
    
    return False


def evaluate_image_pair(original_img, stylized_img, style_text, clip_model, preprocess, lpips_model, device):
    """Evaluate a pair of original and stylized images"""
    
    metrics = {}
    
    # Compute PSNR
    metrics['psnr'] = compute_psnr(original_img, stylized_img)
    
    # Compute LPIPS
    metrics['lpips'] = compute_lpips(original_img, stylized_img, lpips_model, device)
    
    # Compute CLIP scores
    if style_text:
        metrics['clip_directional_similarity'] = compute_clip_directional_similarity(
            original_img, stylized_img, style_text, clip_model, preprocess, device
        )
        metrics['clip_score_original'] = compute_clip_score(
            original_img, style_text, clip_model, preprocess, device
        )
        metrics['clip_score_stylized'] = compute_clip_score(
            stylized_img, style_text, clip_model, preprocess, device
        )
    
    # Compute CLIP consistency (content preservation)
    metrics['clip_consistency'] = compute_clip_consistency(
        original_img, stylized_img, clip_model, preprocess, device
    )
    
    # Compute CLIP-F score (harmonic mean of directional similarity and consistency)
    if 'clip_directional_similarity' in metrics:
        dir_sim = max(0.0, metrics['clip_directional_similarity'])  # Ensure non-negative
        consistency = max(0.0, metrics['clip_consistency'])  # Ensure non-negative
        if dir_sim > 0 and consistency > 0:
            metrics['clip_f_score'] = 2 * (dir_sim * consistency) / (dir_sim + consistency)
        else:
            metrics['clip_f_score'] = 0.0
    
    return metrics


def evaluate_all_images(original_dir, stylized_dir, style_text, device, args):
    """Evaluate all image pairs between original and stylized directories"""
    
    # Load models
    logger.info("Loading evaluation models...")
    clip_model, preprocess = load_clip_model(device)
    lpips_model = load_lpips_model(device)
    
    # Get image files
    original_files = sorted(glob.glob(os.path.join(original_dir, '*.png')))
    stylized_files = sorted(glob.glob(os.path.join(stylized_dir, '*.png')))
    
    if len(original_files) != len(stylized_files):
        logger.warning(f"Mismatch in number of files: {len(original_files)} vs {len(stylized_files)}")
    
    all_metrics = []
    
    logger.info(f"Evaluating {min(len(original_files), len(stylized_files))} image pairs...")
    
    for orig_file, style_file in tqdm(zip(original_files, stylized_files)):
        # Load images
        orig_img = cv2.imread(orig_file)
        style_img = cv2.imread(style_file)
        
        if orig_img is None or style_img is None:
            logger.warning(f"Failed to load images: {orig_file}, {style_file}")
            continue
        
        # Convert BGR to RGB
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        
        # Check if images are side-by-side comparisons and split them
        if args.force_split_sidebyside or is_side_by_side_image(orig_img):
            if args.force_split_sidebyside:
                logger.info(f"Force splitting side-by-side image in original: {orig_file}")
            else:
                logger.info(f"Detected side-by-side image in original: {orig_file}")
            gt_part, rendered_part = split_side_by_side_image(orig_img)
            orig_img = gt_part  # Use ground truth part
        
        if args.force_split_sidebyside or is_side_by_side_image(style_img):
            if args.force_split_sidebyside:
                logger.info(f"Force splitting side-by-side image in stylized: {style_file}")
            else:
                logger.info(f"Detected side-by-side image in stylized: {style_file}")
            gt_part, rendered_part = split_side_by_side_image(style_img)
            style_img = rendered_part  # Use rendered part
        
        # Resize if necessary
        if orig_img.shape != style_img.shape:
            style_img = cv2.resize(style_img, (orig_img.shape[1], orig_img.shape[0]))
        
        # Evaluate this pair
        metrics = evaluate_image_pair(
            orig_img, style_img, style_text, clip_model, preprocess, lpips_model, device
        )
        
        # Add frame info
        metrics['original_file'] = os.path.basename(orig_file)
        metrics['stylized_file'] = os.path.basename(style_file)
        
        all_metrics.append(metrics)
    
    return all_metrics


def compute_summary_metrics(all_metrics):
    """Compute summary statistics from all metrics"""
    
    if not all_metrics:
        return {}
    
    summary = {}
    
    # Get all metric keys (except file names)
    metric_keys = [k for k in all_metrics[0].keys() if not k.endswith('_file')]
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and m[key] != float('inf')]
        
        if values:
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
            summary[f'{key}_median'] = np.median(values)
    
    summary['num_evaluated_pairs'] = len(all_metrics)
    
    return summary


@torch.no_grad()
def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    safe_state(seed=42)
    
    print("=== Stylized Scene Evaluation ===")
    print(f"Original scene checkpoint: {args.original_scene_ckpt}")
    print(f"Stylized scene checkpoint: {args.scene_ckpt}")
    print(f"Human checkpoint: {args.human_ckpt}")
    print(f"Style text: {args.style_text}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    
    # Load configuration
    if os.path.exists(args.cfg_file):
        cfg_file = OmegaConf.load(args.cfg_file)
    else:
        print(f'Config file not found: {args.cfg_file}')
        return
    
    # Merge with default config
    cfg = OmegaConf.merge(default_cfg, cfg_file)
    cfg.eval = True
    cfg.dataset.seq = args.dataset_seq
    cfg.mode = args.mode
    cfg.logdir = args.output_dir
    
    # Setup logger
    get_logger(cfg)
    
    try:
        # Render original scene
        logger.info("Rendering original scene...")
        original_dir = render_scene_with_checkpoint(cfg, args, args.original_scene_ckpt, 'original')
        
        if not original_dir:
            logger.error("Failed to render original scene")
            return
        
        # Render stylized scene
        logger.info("Rendering stylized scene...")
        stylized_dir = render_scene_with_checkpoint(cfg, args, args.scene_ckpt, 'stylized')
        
        if not stylized_dir:
            logger.error("Failed to render stylized scene")
            return
        
        # Evaluate all image pairs
        logger.info("Computing evaluation metrics...")
        all_metrics = evaluate_all_images(original_dir, stylized_dir, args.style_text, device, args)
        
        # Compute summary statistics
        summary_metrics = compute_summary_metrics(all_metrics)
        
        # Save results
        results = {
            'summary_metrics': summary_metrics,
            'per_frame_metrics': all_metrics,
            'evaluation_config': {
                'original_checkpoint': args.original_scene_ckpt,
                'stylized_checkpoint': args.scene_ckpt,
                'human_checkpoint': args.human_ckpt,
                'style_text': args.style_text,
                'mode': args.mode,
                'dataset_seq': args.dataset_seq,
                'num_views': len(all_metrics)
            }
        }
        
        # Save detailed results
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save summary for easy reading
        summary_file = os.path.join(args.output_dir, 'summary_metrics.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_metrics, f, indent=4)
        
        # Print summary
        print(f"\n=== Evaluation Complete ===")
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        print(f"\n=== Summary Metrics ===")
        
        for key, value in summary_metrics.items():
            if not key.endswith(('_std', '_min', '_max', '_median')) and key != 'num_evaluated_pairs':
                print(f"{key}: {value:.4f}")
        
        print(f"Evaluated {summary_metrics.get('num_evaluated_pairs', 0)} image pairs")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()