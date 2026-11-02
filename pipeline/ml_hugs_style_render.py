#!/usr/bin/env python3
"""
ML-HUGS Style Transfer Rendering Script
Renders scenes using stylized scene models from style transfer training.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

# Add the repository and external ML-HUGS checkout to the import path.
repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))
from runtime import configure_runtime

configure_runtime()

# Import ml-hugs modules
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models import SceneGS
from hugs.renderer.gs_renderer import render_human_scene
from hugs.datasets import NeumanDataset
from hugs.datasets.utils import get_smpl_static_params, get_rotating_camera
from hugs.utils.image import save_image
from hugs.utils.general import safe_state


class StyleTransferRenderer:
    """Renderer for stylized ML-HUGS models"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.human_gs = None
        self.scene_gs = None
        self.dataset = None
        
        # Background
        self.background_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
    
    def load_human_model(self, checkpoint_path):
        """Load human model from checkpoint"""
        print(f"Loading human model from: {checkpoint_path}")
        
        # Find human checkpoint file
        if os.path.isdir(checkpoint_path):
            human_ckpt_files = glob.glob(os.path.join(checkpoint_path, "*human*.pth"))
            human_ckpt_files += glob.glob(os.path.join(checkpoint_path, "ckpt", "*human*.pth"))
            if not human_ckpt_files:
                raise FileNotFoundError(f"No human checkpoint found in {checkpoint_path}")
            human_ckpt_path = sorted(human_ckpt_files)[-1]
        else:
            human_ckpt_path = checkpoint_path
        
        # Load checkpoint
        human_state = torch.load(human_ckpt_path, map_location=self.device, weights_only=False)
        
        # Create human model
        dummy_betas = torch.zeros(10, device=self.device)
        self.human_gs = HUGS_TRIMLP(
            sh_degree=3,
            n_subdivision=0,
            use_surface=False,
            init_2d=False,
            rotate_sh=True,
            isotropic=False,
            init_scale_multiplier=1.0,
            n_features=32,
            use_deformer=True,
            disable_posedirs=False,
            triplane_res=256,
            betas=dummy_betas
        )
        
        # Restore human model state
        self.restore_human_model_state(human_state)
        self.human_gs.eval()
        
        print(f"Human model loaded successfully with {self.human_gs.n_gs} Gaussians")
    
    def restore_human_model_state(self, human_state):
        """Restore human model state from checkpoint"""
        if 'active_sh_degree' in human_state:
            self.human_gs.active_sh_degree = human_state['active_sh_degree']
        
        if 'xyz' in human_state:
            self.human_gs._xyz = human_state['xyz']
        
        if 'triplane' in human_state:
            self.human_gs.triplane.load_state_dict(human_state['triplane'])
        
        if 'appearance_dec' in human_state:
            self.human_gs.appearance_dec.load_state_dict(human_state['appearance_dec'])
        
        if 'geometry_dec' in human_state:
            self.human_gs.geometry_dec.load_state_dict(human_state['geometry_dec'])
        
        if 'deformation_dec' in human_state:
            self.human_gs.deformation_dec.load_state_dict(human_state['deformation_dec'], strict=False)
        
        # Restore other state variables
        state_vars = ['scaling_multiplier', 'max_radii2D', 'xyz_gradient_accum', 'denom', 'spatial_lr_scale']
        for var in state_vars:
            if var in human_state:
                setattr(self.human_gs, var, human_state[var])
    
    def load_stylized_scene_model(self, checkpoint_path):
        """Load stylized scene model from checkpoint"""
        print(f"Loading stylized scene model from: {checkpoint_path}")
        
        # Load checkpoint
        scene_state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Create scene model
        self.scene_gs = SceneGS(sh_degree=3)
        
        # Restore scene model state
        self.scene_gs._xyz = scene_state['_xyz']
        self.scene_gs._features_dc = scene_state['_features_dc']
        self.scene_gs._features_rest = scene_state['_features_rest']
        self.scene_gs._scaling = scene_state['_scaling']
        self.scene_gs._rotation = scene_state['_rotation']
        self.scene_gs._opacity = scene_state['_opacity']
        self.scene_gs.active_sh_degree = scene_state['active_sh_degree']
        self.scene_gs.max_radii2D = scene_state['max_radii2D']
        self.scene_gs.xyz_gradient_accum = scene_state['xyz_gradient_accum']
        self.scene_gs.denom = scene_state['denom']
        
        self.scene_gs.eval()
        
        print(f"Stylized scene model loaded successfully with {self.scene_gs._xyz.shape[0]} Gaussians")
        
        # Print style information if available
        if 'style_prompt' in scene_state:
            print(f"Style: {scene_state['style_prompt']}")
    
    def load_dataset(self, dataset_path, sequence_name):
        """Load dataset"""
        print(f"Loading dataset from: {dataset_path}")
        
        # Create a minimal config for dataset loading
        cfg = type('Config', (), {})()
        cfg.dataset = type('Dataset', (), {})()
        cfg.dataset.seq = sequence_name
        cfg.dataset.data_dir = dataset_path
        cfg.dataset.img_res = [512, 512]
        cfg.train = type('Train', (), {})()
        cfg.train.batch_size = 1
        
        self.dataset = NeumanDataset(cfg, split=self.args.split)
        print(f"Dataset loaded with {len(self.dataset)} views")
    
    def render_sequence(self, output_dir, mode='human_scene'):
        """Render the entire sequence"""
        print(f"Rendering sequence in mode: {mode}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for i in tqdm(range(len(self.dataset)), desc="Rendering"):
                data = self.dataset[i]
                
                # Get human output
                static_smpl_params = get_smpl_static_params(
                    betas=self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else torch.zeros(10, device=self.device),
                    pose_type='a_pose'
                )
                
                human_gs_out = self.human_gs.forward(
                    global_orient=data.get('global_orient', static_smpl_params['global_orient']),
                    body_pose=data.get('body_pose', static_smpl_params['body_pose']),
                    betas=data.get('betas', static_smpl_params['betas']),
                    transl=data.get('transl', static_smpl_params['transl']),
                    smpl_scale=data.get('smpl_scale', static_smpl_params['smpl_scale']),
                    dataset_idx=i,
                    is_train=False,
                    ext_tfs=data.get('ext_tfs', None)
                )
                
                # Get scene output
                scene_gs_out = self.scene_gs.forward()
                
                # Render
                render_pkg = render_human_scene(
                    data=data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    bg_color=self.background_color,
                    render_mode=mode
                )
                
                # Save image
                rendered_image = render_pkg["render"]
                output_path = os.path.join(output_dir, f"frame_{i:06d}.png")
                save_image(rendered_image, output_path)
        
        print(f"Sequence rendered to: {output_dir}")
    
    def render_comparison(self, output_dir):
        """Render comparison between human, scene, and combined"""
        print("Rendering comparison views")
        
        comparison_dir = os.path.join(output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Render different modes
        modes = ['human', 'scene', 'human_scene']
        
        for mode in modes:
            mode_dir = os.path.join(comparison_dir, mode)
            self.render_sequence(mode_dir, mode=mode)
    
    def render_rotating_sequence(self, output_dir, n_frames=120):
        """Render rotating camera sequence"""
        print(f"Rendering rotating sequence with {n_frames} frames")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Use first frame data as reference
        ref_data = self.dataset[0]
        
        with torch.no_grad():
            for i in tqdm(range(n_frames), desc="Rendering rotation"):
                # Get rotating camera
                angle = (i / n_frames) * 2 * np.pi
                cam_data = get_rotating_camera(ref_data, angle, radius=2.0)
                
                # Get human output (static pose for rotation)
                static_smpl_params = get_smpl_static_params(
                    betas=self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else torch.zeros(10, device=self.device),
                    pose_type='a_pose'
                )
                
                human_gs_out = self.human_gs.forward(
                    global_orient=static_smpl_params['global_orient'],
                    body_pose=static_smpl_params['body_pose'],
                    betas=static_smpl_params['betas'],
                    transl=static_smpl_params['transl'],
                    smpl_scale=static_smpl_params['smpl_scale'],
                    dataset_idx=0,
                    is_train=False,
                    ext_tfs=None
                )
                
                # Get scene output
                scene_gs_out = self.scene_gs.forward()
                
                # Render
                render_pkg = render_human_scene(
                    data=cam_data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    bg_color=self.background_color,
                    render_mode='human_scene'
                )
                
                # Save image
                rendered_image = render_pkg["render"]
                output_path = os.path.join(output_dir, f"rotate_{i:06d}.png")
                save_image(rendered_image, output_path)
        
        print(f"Rotating sequence rendered to: {output_dir}")
    
    def create_video(self, image_dir, output_path, fps=30):
        """Create video from rendered images"""
        try:
            import cv2
            
            # Get image files
            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
            if not image_files:
                print("No images found for video creation")
                return
            
            # Read first image to get dimensions
            first_img = cv2.imread(os.path.join(image_dir, image_files[0]))
            h, w, c = first_img.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # Add images to video
            for img_file in tqdm(image_files, desc="Creating video"):
                img_path = os.path.join(image_dir, img_file)
                img = cv2.imread(img_path)
                video_writer.write(img)
            
            video_writer.release()
            print(f"Video created: {output_path}")
            
        except ImportError:
            print("OpenCV not available. Skipping video creation.")
            print(f"You can create a video manually from images in: {image_dir}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML-HUGS Style Transfer Rendering")
    
    # Model paths
    parser.add_argument("--human_checkpoint", required=True,
                       help="Path to pre-trained human model checkpoint")
    parser.add_argument("--stylized_scene_checkpoint", required=True,
                       help="Path to stylized scene model checkpoint")
    parser.add_argument("--dataset_path", required=True,
                       help="Path to dataset")
    parser.add_argument("--sequence", required=True,
                       help="Sequence name")
    
    # Rendering options
    parser.add_argument("--split", choices=['train', 'test', 'val'], default='test',
                       help="Dataset split to render")
    parser.add_argument("--render_mode", choices=['human', 'scene', 'human_scene'], 
                       default='human_scene',
                       help="Rendering mode")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for rendered images")
    
    # Special rendering modes
    parser.add_argument("--render_comparison", action="store_true",
                       help="Render comparison between different modes")
    parser.add_argument("--render_rotation", action="store_true",
                       help="Render rotating camera sequence")
    parser.add_argument("--rotation_frames", type=int, default=120,
                       help="Number of frames for rotation sequence")
    
    # Video creation
    parser.add_argument("--create_video", action="store_true",
                       help="Create video from rendered images")
    parser.add_argument("--fps", type=int, default=30,
                       help="FPS for video creation")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize renderer
    renderer = StyleTransferRenderer(args)
    
    # Load models and data
    renderer.load_human_model(args.human_checkpoint)
    renderer.load_stylized_scene_model(args.stylized_scene_checkpoint)
    renderer.load_dataset(args.dataset_path, args.sequence)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Render based on options
    if args.render_comparison:
        renderer.render_comparison(args.output_dir)
    elif args.render_rotation:
        rotation_dir = os.path.join(args.output_dir, "rotation")
        renderer.render_rotating_sequence(rotation_dir, args.rotation_frames)
        
        if args.create_video:
            video_path = os.path.join(args.output_dir, "rotation.mp4")
            renderer.create_video(rotation_dir, video_path, args.fps)
    else:
        # Regular sequence rendering
        sequence_dir = os.path.join(args.output_dir, args.render_mode)
        renderer.render_sequence(sequence_dir, args.render_mode)
        
        if args.create_video:
            video_path = os.path.join(args.output_dir, f"{args.render_mode}.mp4")
            renderer.create_video(sequence_dir, video_path, args.fps)
    
    print("Rendering completed successfully!")


if __name__ == "__main__":
    main()