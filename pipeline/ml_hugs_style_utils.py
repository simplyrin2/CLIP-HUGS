#!/usr/bin/env python3
"""
ML-HUGS Style Transfer Utility Script
Helper utilities for setting up and running style transfer experiments.
"""

import os
import sys
import argparse
import glob
import json
from pathlib import Path

import torch


def find_model_checkpoints(base_dir):
    """Find human and scene model checkpoints in a directory"""
    human_ckpts = []
    scene_ckpts = []
    
    # Search patterns
    patterns = [
        "*human*.pth",
        "ckpt/*human*.pth",
        "*scene*.pth", 
        "ckpt/*scene*.pth"
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(base_dir, pattern))
        for match in matches:
            if "human" in os.path.basename(match):
                human_ckpts.append(match)
            elif "scene" in os.path.basename(match):
                scene_ckpts.append(match)
    
    return sorted(human_ckpts), sorted(scene_ckpts)


def inspect_checkpoint(ckpt_path):
    """Inspect checkpoint contents"""
    print(f"\nInspecting checkpoint: {ckpt_path}")
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        
        # Check for style information
        if 'style_prompt' in ckpt:
            print(f"Style prompt: {ckpt['style_prompt']}")
        
        if 'iteration' in ckpt:
            print(f"Training iteration: {ckpt['iteration']}")
        
        # Check tensor shapes
        for key, value in ckpt.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif key in ['active_sh_degree', 'iteration']:
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


def create_experiment_config(args):
    """Create experiment configuration file"""
    config = {
        "experiment_name": args.experiment_name,
        "human_checkpoint": args.human_checkpoint,
        "scene_checkpoint": args.scene_checkpoint,
        "dataset_path": args.dataset_path,
        "sequence": args.sequence,
        "style_prompt": args.style_prompt,
        "style_image": args.style_image,
        "iterations": args.iterations,
        "loss_weights": {
            "content": args.content_weight,
            "patch": args.patch_weight,
            "directional": args.directional_weight,
            "background": args.background_weight
        },
        "learning_rates": {
            "xyz": args.lr_xyz,
            "feature": args.lr_feature,
            "opacity": args.lr_opacity,
            "scaling": args.lr_scaling,
            "rotation": args.lr_rotation
        }
    }
    
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment config saved to: {config_path}")
    return config_path


def generate_training_command(args):
    """Generate training command"""
    cmd_parts = [
        "python ml_hugs_style_transfer.py",
        f"--human_checkpoint {args.human_checkpoint}",
        f"--scene_checkpoint {args.scene_checkpoint}",
        f"--dataset_path {args.dataset_path}",
        f"--sequence {args.sequence}",
        f"--style_prompt '{args.style_prompt}'"
    ]
    
    if args.style_image:
        cmd_parts.append(f"--style_image {args.style_image}")
    
    if args.output_dir:
        cmd_parts.append(f"--output_dir {args.output_dir}")
    
    cmd_parts.extend([
        f"--iterations {args.iterations}",
        f"--content_weight {args.content_weight}",
        f"--patch_weight {args.patch_weight}",
        f"--directional_weight {args.directional_weight}",
        f"--background_weight {args.background_weight}"
    ])
    
    if args.render_test:
        cmd_parts.append("--render_test")
    
    return " \\\n  ".join(cmd_parts)


def generate_rendering_command(args):
    """Generate rendering command"""
    # Find stylized checkpoint
    stylized_ckpt = None
    if args.stylized_checkpoint:
        stylized_ckpt = args.stylized_checkpoint
    elif args.output_dir:
        # Look for final checkpoint
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            final_ckpts = glob.glob(os.path.join(ckpt_dir, "*final*.pth"))
            if final_ckpts:
                stylized_ckpt = sorted(final_ckpts)[-1]
    
    if not stylized_ckpt:
        print("Error: No stylized checkpoint found")
        return None
    
    render_output = args.render_output_dir or os.path.join(args.output_dir, "renders")
    
    cmd_parts = [
        "python ml_hugs_style_render.py",
        f"--human_checkpoint {args.human_checkpoint}",
        f"--stylized_scene_checkpoint {stylized_ckpt}",
        f"--dataset_path {args.dataset_path}",
        f"--sequence {args.sequence}",
        f"--output_dir {render_output}"
    ]
    
    if args.render_comparison:
        cmd_parts.append("--render_comparison")
    
    if args.render_rotation:
        cmd_parts.append("--render_rotation")
    
    if args.create_video:
        cmd_parts.append("--create_video")
    
    return " \\\n  ".join(cmd_parts)


def list_available_models(base_dir):
    """List available trained models"""
    print(f"\nSearching for models in: {base_dir}")
    
    model_dirs = []
    for root, dirs, files in os.walk(base_dir):
        # Look for directories with checkpoints
        has_human = any("human" in f for f in files if f.endswith('.pth'))
        has_scene = any("scene" in f for f in files if f.endswith('.pth'))
        
        if has_human or has_scene:
            rel_path = os.path.relpath(root, base_dir)
            model_dirs.append((rel_path, has_human, has_scene))
    
    if model_dirs:
        print("\nAvailable models:")
        for model_dir, has_human, has_scene in model_dirs:
            status = []
            if has_human:
                status.append("Human")
            if has_scene:
                status.append("Scene")
            print(f"  {model_dir} - {', '.join(status)}")
    else:
        print("No models found")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML-HUGS Style Transfer Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Find models command
    find_parser = subparsers.add_parser('find-models', help='Find available model checkpoints')
    find_parser.add_argument('base_dir', help='Base directory to search for models')
    
    # Inspect checkpoint command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect checkpoint contents')
    inspect_parser.add_argument('checkpoint_path', help='Path to checkpoint file')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available trained models')
    list_parser.add_argument('base_dir', help='Base directory to search for models')
    
    # Generate training command
    train_cmd_parser = subparsers.add_parser('gen-train-cmd', help='Generate training command')
    train_cmd_parser.add_argument('--human_checkpoint', required=True)
    train_cmd_parser.add_argument('--scene_checkpoint', required=True)
    train_cmd_parser.add_argument('--dataset_path', required=True)
    train_cmd_parser.add_argument('--sequence', required=True)
    train_cmd_parser.add_argument('--style_prompt', required=True)
    train_cmd_parser.add_argument('--style_image', default=None)
    train_cmd_parser.add_argument('--output_dir', default=None)
    train_cmd_parser.add_argument('--iterations', type=int, default=10000)
    train_cmd_parser.add_argument('--content_weight', type=float, default=1.0)
    train_cmd_parser.add_argument('--patch_weight', type=float, default=5000.0)
    train_cmd_parser.add_argument('--directional_weight', type=float, default=2500.0)
    train_cmd_parser.add_argument('--background_weight', type=float, default=100.0)
    train_cmd_parser.add_argument('--lr_xyz', type=float, default=1.6e-4)
    train_cmd_parser.add_argument('--lr_feature', type=float, default=2.5e-3)
    train_cmd_parser.add_argument('--lr_opacity', type=float, default=5.0e-2)
    train_cmd_parser.add_argument('--lr_scaling', type=float, default=5.0e-3)
    train_cmd_parser.add_argument('--lr_rotation', type=float, default=1.0e-3)
    train_cmd_parser.add_argument('--render_test', action='store_true')
    train_cmd_parser.add_argument('--experiment_name', default='style_transfer_experiment')
    
    # Generate rendering command
    render_cmd_parser = subparsers.add_parser('gen-render-cmd', help='Generate rendering command')
    render_cmd_parser.add_argument('--human_checkpoint', required=True)
    render_cmd_parser.add_argument('--stylized_checkpoint', default=None)
    render_cmd_parser.add_argument('--dataset_path', required=True)
    render_cmd_parser.add_argument('--sequence', required=True)
    render_cmd_parser.add_argument('--output_dir', default=None)
    render_cmd_parser.add_argument('--render_output_dir', default=None)
    render_cmd_parser.add_argument('--render_comparison', action='store_true')
    render_cmd_parser.add_argument('--render_rotation', action='store_true')
    render_cmd_parser.add_argument('--create_video', action='store_true')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    if args.command == 'find-models':
        human_ckpts, scene_ckpts = find_model_checkpoints(args.base_dir)
        
        print(f"\nFound {len(human_ckpts)} human checkpoints:")
        for ckpt in human_ckpts:
            print(f"  {ckpt}")
        
        print(f"\nFound {len(scene_ckpts)} scene checkpoints:")
        for ckpt in scene_ckpts:
            print(f"  {ckpt}")
    
    elif args.command == 'inspect':
        inspect_checkpoint(args.checkpoint_path)
    
    elif args.command == 'list-models':
        list_available_models(args.base_dir)
    
    elif args.command == 'gen-train-cmd':
        # Create config file
        if args.output_dir:
            create_experiment_config(args)
        
        # Generate command
        cmd = generate_training_command(args)
        print("\nTraining command:")
        print(cmd)
        
        # Save to file
        if args.output_dir:
            cmd_file = os.path.join(args.output_dir, "train_command.sh")
            with open(cmd_file, 'w') as f:
                f.write("#!/bin/bash\n\n")
                f.write(cmd + "\n")
            os.chmod(cmd_file, 0o755)
            print(f"\nCommand saved to: {cmd_file}")
    
    elif args.command == 'gen-render-cmd':
        cmd = generate_rendering_command(args)
        if cmd:
            print("\nRendering command:")
            print(cmd)
            
            # Save to file
            if args.output_dir:
                cmd_file = os.path.join(args.output_dir, "render_command.sh")
                os.makedirs(args.output_dir, exist_ok=True)
                with open(cmd_file, 'w') as f:
                    f.write("#!/bin/bash\n\n")
                    f.write(cmd + "\n")
                os.chmod(cmd_file, 0o755)
                print(f"\nCommand saved to: {cmd_file}")
    
    else:
        print("Please specify a command. Use -h for help.")


if __name__ == "__main__":
    main()