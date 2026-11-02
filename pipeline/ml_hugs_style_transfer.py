#!/usr/bin/env python3
"""
ML-HUGS Style Transfer Training Script
Applies CLIP Gaussian losses to stylize scene components while keeping human unchanged.
Uses the losses from CLIPGaussian/3D to change scene styles in ml-hugs output.
"""

import os
import sys
import argparse
import glob
import uuid
from pathlib import Path
from random import randint
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np

# Add paths for the repository and the external ML-HUGS checkout.
repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))
from runtime import configure_runtime

configure_runtime()

# Import ml-hugs modules
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.models import SceneGS
from hugs.renderer.gs_renderer import render_human_scene, render
from hugs.datasets import NeumanDataset
from hugs.datasets.utils import get_smpl_static_params
from hugs.cfg.config import cfg as default_cfg
from hugs.utils.general import safe_state
from hugs.utils.image import save_image, psnr
from hugs.losses.utils import ssim

# Import CLIPGaussian modules
import CLIP
from CLIP.utils.image_utils import img_normalize, clip_normalize
from CLIP.scene.VGG import get_features

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class StyleTransferTrainer:
    """Main trainer class for ML-HUGS style transfer"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.human_gs = None
        self.scene_gs = None
        self.dataset = None
        
        # Load CLIP and VGG models for style transfer
        self.clip_model = CLIP.load_model()
        self.vgg_model = models.vgg19(weights='DEFAULT').features.to(self.device)
        
        # Style embedding
        self.style_direction = None
        
        # Training setup
        self.background_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.crop_augment = transforms.Compose([
            transforms.RandomCrop(args.crop_size)
        ])
        self.perspective_augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
            transforms.Resize(224)
        ])
        
        # Setup output directory
        self.setup_output_directory()
        
    def setup_output_directory(self):
        """Setup output directory and logging"""
        if not self.args.output_dir:
            unique_str = str(uuid.uuid4())
            self.args.output_dir = os.path.join("./output_style_transfer", unique_str[:10])
        
        os.makedirs(self.args.output_dir, exist_ok=True)
        print(f"Output directory: {self.args.output_dir}")
        
        # Setup tensorboard
        self.tb_writer = None
        if TENSORBOARD_FOUND:
            self.tb_writer = SummaryWriter(self.args.output_dir)
    
    def load_human_model(self, checkpoint_path):
        """Load pre-trained human model from checkpoint"""
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
        self.human_gs.eval()  # Keep human model in eval mode (frozen)
        
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
        
        # Restore optimizer state variables if available
        state_vars = ['scaling_multiplier', 'max_radii2D', 'xyz_gradient_accum', 'denom', 'spatial_lr_scale']
        for var in state_vars:
            if var in human_state:
                setattr(self.human_gs, var, human_state[var])
    
    def load_scene_model(self, checkpoint_path):
        """Load pre-trained scene model from checkpoint"""
        print(f"Loading scene model from: {checkpoint_path}")
        
        # Find scene checkpoint file
        if os.path.isdir(checkpoint_path):
            scene_ckpt_files = glob.glob(os.path.join(checkpoint_path, "*scene*.pth"))
            scene_ckpt_files += glob.glob(os.path.join(checkpoint_path, "ckpt", "*scene*.pth"))
            if not scene_ckpt_files:
                raise FileNotFoundError(f"No scene checkpoint found in {checkpoint_path}")
            scene_ckpt_path = sorted(scene_ckpt_files)[-1]
        else:
            scene_ckpt_path = checkpoint_path
        
        # Load checkpoint
        scene_state = torch.load(scene_ckpt_path, map_location=self.device, weights_only=False)
        
        # Create scene model
        self.scene_gs = SceneGS(sh_degree=3)
        
        # Restore scene model state
        self.scene_gs._xyz = scene_state['_xyz'] if '_xyz' in scene_state else scene_state['xyz']
        self.scene_gs._features_dc = scene_state['_features_dc'] if '_features_dc' in scene_state else scene_state['features_dc']
        self.scene_gs._features_rest = scene_state['_features_rest'] if '_features_rest' in scene_state else scene_state['features_rest']
        self.scene_gs._scaling = scene_state['_scaling'] if '_scaling' in scene_state else scene_state['scaling']
        self.scene_gs._rotation = scene_state['_rotation'] if '_rotation' in scene_state else scene_state['rotation']
        self.scene_gs._opacity = scene_state['_opacity'] if '_opacity' in scene_state else scene_state['opacity']
        self.scene_gs.active_sh_degree = scene_state['active_sh_degree']
        self.scene_gs.max_radii2D = scene_state['max_radii2D']
        self.scene_gs.xyz_gradient_accum = scene_state['xyz_gradient_accum']
        self.scene_gs.denom = scene_state['denom']
        
        # Setup scene model for training (only scene will be optimized)
        self.setup_scene_optimizer()
        
        print(f"Scene model loaded successfully with {self.scene_gs._xyz.shape[0]} Gaussians")
    
    def setup_scene_optimizer(self):
        """Setup optimizer for scene model only"""
        l = [
            {'params': [self.scene_gs._xyz], 'lr': self.args.lr_xyz, "name": "xyz"},
            {'params': [self.scene_gs._features_dc], 'lr': self.args.lr_feature, "name": "f_dc"},
            {'params': [self.scene_gs._features_rest], 'lr': self.args.lr_feature / 20.0, "name": "f_rest"},
            {'params': [self.scene_gs._opacity], 'lr': self.args.lr_opacity, "name": "opacity"},
            {'params': [self.scene_gs._scaling], 'lr': self.args.lr_scaling, "name": "scaling"},
            {'params': [self.scene_gs._rotation], 'lr': self.args.lr_rotation, "name": "rotation"}
        ]
        self.scene_gs.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def load_dataset(self, dataset_path, sequence_name):
        """Load dataset for training views"""
        print(f"Loading dataset from: {dataset_path}")
        
        # Create a minimal config for dataset loading
        cfg = type('Config', (), {})()
        cfg.dataset = type('Dataset', (), {})()
        cfg.dataset.seq = sequence_name
        cfg.dataset.data_dir = dataset_path
        cfg.dataset.img_res = [512, 512]  # Default resolution
        cfg.train = type('Train', (), {})()
        cfg.train.batch_size = 1
        
        self.dataset = NeumanDataset(cfg, split='train')
        print(f"Dataset loaded with {len(self.dataset)} training views")
    
    def setup_style_embedding(self):
        """Setup style embedding from text prompt or image"""
        print(f"Setting up style embedding for: {self.args.style_prompt}")
        
        self.style_direction = CLIP.get_style_embedding(
            self.clip_model,
            self.args.style_prompt,
            self.args.style_image,
            self.args.object_prompt
        )
        print("Style embedding computed successfully")
    
    def precompute_view_features(self):
        """Precompute CLIP and VGG features for all training views"""
        print("Precomputing view features...")
        
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            gt_image = data['rgb'].to(self.device)
            
            # Compute CLIP features
            with torch.no_grad():
                clip_features = self.clip_model.encode_image(
                    clip_normalize(gt_image.unsqueeze(0))
                )
                clip_features /= clip_features.norm(dim=-1, keepdim=True)
                data['clip_features'] = clip_features
                
                # Compute VGG features
                vgg_features = get_features(img_normalize(gt_image), self.vgg_model)
                data['vgg_features'] = vgg_features
        
        print("View features precomputed successfully")
    
    def compute_style_losses(self, rendered_image, gt_image, gt_clip_features, gt_vgg_features):
        """Compute all style transfer losses"""
        losses = {}
        
        # VGG Content Loss
        rendered_features = get_features(img_normalize(rendered_image), self.vgg_model)
        content_loss = 0
        content_loss += torch.mean((gt_vgg_features['conv4_2'] - rendered_features['conv4_2']) ** 2)
        content_loss += torch.mean((gt_vgg_features['conv5_2'] - rendered_features['conv5_2']) ** 2)
        losses['content'] = content_loss
        
        # Patch CLIP Loss
        img_crops = [
            self.perspective_augment(self.crop_augment(rendered_image.unsqueeze(0)))
            for _ in range(self.args.num_crops)
        ]
        img_crops = torch.cat(img_crops, dim=0)
        
        crop_features = self.clip_model.encode_image(clip_normalize(img_crops))
        crop_features /= crop_features.norm(dim=-1, keepdim=True)
        
        crop_directions = crop_features - gt_clip_features
        crop_directions /= crop_directions.norm(dim=-1, keepdim=True)
        
        patch_loss = (1 - torch.cosine_similarity(
            crop_directions,
            self.style_direction.repeat(crop_features.size(0), 1),
            dim=1
        )).mean()
        losses['patch'] = patch_loss
        
        # Directional CLIP Loss
        rendered_clip_features = self.clip_model.encode_image(
            clip_normalize(rendered_image.unsqueeze(0))
        )
        rendered_clip_features /= rendered_clip_features.norm(dim=-1, keepdim=True)
        
        direction = rendered_clip_features - gt_clip_features
        direction /= direction.norm(dim=-1, keepdim=True)
        
        directional_loss = (1 - torch.cosine_similarity(
            direction,
            self.style_direction.repeat(rendered_clip_features.size(0), 1),
            dim=1
        )).mean()
        losses['directional'] = directional_loss
        
        # Background Loss (optional)
        if self.args.background_weight > 0:
            # Simple background regularization
            bg_loss = torch.mean((rendered_image - self.background_color.view(-1, 1, 1)) ** 2)
            losses['background'] = bg_loss
        else:
            losses['background'] = torch.tensor(0.0, device=self.device)
        
        return losses
    
    def train_step(self, data_idx):
        """Single training step"""
        data = self.dataset[data_idx]
        
        # Get human output (frozen)
        with torch.no_grad():
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
                dataset_idx=data_idx,
                is_train=False,
                ext_tfs=data.get('ext_tfs', None)
            )
        
        # Get scene output (trainable)
        scene_gs_out = self.scene_gs.forward()
        
        # Random background for training
        if self.args.random_background:
            bg_color = torch.rand(3, device=self.device)
        else:
            bg_color = self.background_color
        
        # Render combined human+scene
        render_pkg = render_human_scene(
            data=data,
            human_gs_out=human_gs_out,
            scene_gs_out=scene_gs_out,
            bg_color=bg_color,
            render_mode='human_scene'
        )
        
        rendered_image = render_pkg["render"]
        gt_image = data['rgb'].to(self.device)
        
        # Compute losses
        losses = self.compute_style_losses(
            rendered_image,
            gt_image,
            data['clip_features'],
            data['vgg_features']
        )
        
        # Combine losses
        total_loss = (
            self.args.content_weight * losses['content'] +
            self.args.patch_weight * losses['patch'] +
            self.args.directional_weight * losses['directional'] +
            self.args.background_weight * losses['background']
        )
        
        return total_loss, losses, rendered_image
    
    def train(self):
        """Main training loop"""
        print("Starting style transfer training...")
        
        # Load models and data
        self.load_human_model(self.args.human_checkpoint)
        self.load_scene_model(self.args.scene_checkpoint)
        self.load_dataset(self.args.dataset_path, self.args.sequence)
        self.setup_style_embedding()
        self.precompute_view_features()
        
        # Training loop
        progress_bar = tqdm(range(self.args.iterations), desc="Training progress")
        ema_loss = 0.0
        
        for iteration in range(1, self.args.iterations + 1):
            # Random view selection
            data_idx = randint(0, len(self.dataset) - 1)
            
            # Training step
            total_loss, losses, rendered_image = self.train_step(data_idx)
            
            # Backward pass (only scene parameters will be updated)
            self.scene_gs.optimizer.zero_grad()
            total_loss.backward()
            self.scene_gs.optimizer.step()
            
            # Update EMA loss
            ema_loss = 0.4 * total_loss.item() + 0.6 * ema_loss
            
            # Logging
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Total": f"{ema_loss:.6f}",
                    "Content": f"{losses['content'].item():.6f}",
                    "Patch": f"{losses['patch'].item():.6f}",
                    "Dir": f"{losses['directional'].item():.6f}"
                })
                progress_bar.update(10)
            
            # Tensorboard logging
            if self.tb_writer and iteration % 50 == 0:
                self.tb_writer.add_scalar('Loss/Total', total_loss.item(), iteration)
                self.tb_writer.add_scalar('Loss/Content', losses['content'].item(), iteration)
                self.tb_writer.add_scalar('Loss/Patch', losses['patch'].item(), iteration)
                self.tb_writer.add_scalar('Loss/Directional', losses['directional'].item(), iteration)
                self.tb_writer.add_scalar('Loss/Background', losses['background'].item(), iteration)
            
            # Save checkpoints
            if iteration in self.args.save_iterations:
                self.save_checkpoint(iteration)
            
            # Save preview images
            if iteration % self.args.preview_interval == 0:
                self.save_preview_images(iteration, rendered_image, data_idx)
        
        progress_bar.close()
        print("Training completed!")
        
        # Final checkpoint
        self.save_checkpoint(self.args.iterations, final=True)
    
    def save_checkpoint(self, iteration, final=False):
        """Save model checkpoints"""
        checkpoint_dir = os.path.join(self.args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save scene model (only this was trained)
        if final:
            scene_path = os.path.join(checkpoint_dir, "scene_stylized_final.pth")
        else:
            scene_path = os.path.join(checkpoint_dir, f"scene_stylized_{iteration:06d}.pth")
        
        torch.save({
            '_xyz': self.scene_gs._xyz,
            '_features_dc': self.scene_gs._features_dc,
            '_features_rest': self.scene_gs._features_rest,
            '_scaling': self.scene_gs._scaling,
            '_rotation': self.scene_gs._rotation,
            '_opacity': self.scene_gs._opacity,
            'active_sh_degree': self.scene_gs.active_sh_degree,
            'max_radii2D': self.scene_gs.max_radii2D,
            'xyz_gradient_accum': self.scene_gs.xyz_gradient_accum,
            'denom': self.scene_gs.denom,
            'optimizer': self.scene_gs.optimizer.state_dict(),
            'iteration': iteration,
            'style_prompt': self.args.style_prompt
        }, scene_path)
        
        print(f"Checkpoint saved: {scene_path}")
    
    def save_preview_images(self, iteration, rendered_image, data_idx):
        """Save preview images during training"""
        preview_dir = os.path.join(self.args.output_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        
        # Save rendered image
        rendered_path = os.path.join(preview_dir, f"rendered_{iteration:06d}_{data_idx:03d}.png")
        save_image(rendered_image, rendered_path)
        
        # Save GT for comparison
        if iteration == self.args.preview_interval:
            gt_image = self.dataset[data_idx]['rgb']
            gt_path = os.path.join(preview_dir, f"gt_{data_idx:03d}.png")
            save_image(gt_image, gt_path)
    
    def render_test_sequence(self):
        """Render test sequence with stylized scene"""
        print("Rendering test sequence...")
        
        test_dir = os.path.join(self.args.output_dir, "test_renders")
        os.makedirs(test_dir, exist_ok=True)
        
        # Use test split if available, otherwise use validation views from train split
        try:
            test_dataset = NeumanDataset(self.dataset.cfg, split='test')
        except:
            # Use every 10th training view as test
            test_indices = list(range(0, len(self.dataset), 10))
            test_dataset = [self.dataset[i] for i in test_indices]
        
        with torch.no_grad():
            for i, data in enumerate(test_dataset):
                if isinstance(data, dict):
                    test_data = data
                else:
                    test_data = test_dataset[i]
                
                # Get human output
                static_smpl_params = get_smpl_static_params(
                    betas=self.human_gs.betas.detach() if hasattr(self.human_gs, 'betas') else torch.zeros(10, device=self.device),
                    pose_type='a_pose'
                )
                
                human_gs_out = self.human_gs.forward(
                    global_orient=test_data.get('global_orient', static_smpl_params['global_orient']),
                    body_pose=test_data.get('body_pose', static_smpl_params['body_pose']),
                    betas=test_data.get('betas', static_smpl_params['betas']),
                    transl=test_data.get('transl', static_smpl_params['transl']),
                    smpl_scale=test_data.get('smpl_scale', static_smpl_params['smpl_scale']),
                    dataset_idx=i,
                    is_train=False,
                    ext_tfs=test_data.get('ext_tfs', None)
                )
                
                # Get scene output
                scene_gs_out = self.scene_gs.forward()
                
                # Render
                render_pkg = render_human_scene(
                    data=test_data,
                    human_gs_out=human_gs_out,
                    scene_gs_out=scene_gs_out,
                    bg_color=self.background_color,
                    render_mode='human_scene'
                )
                
                # Save image
                rendered_image = render_pkg["render"]
                output_path = os.path.join(test_dir, f"stylized_{i:04d}.png")
                save_image(rendered_image, output_path)
        
        print(f"Test sequence rendered to: {test_dir}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML-HUGS Style Transfer Training")
    
    # Model paths
    parser.add_argument("--human_checkpoint", required=True, 
                       help="Path to pre-trained human model checkpoint")
    parser.add_argument("--scene_checkpoint", required=True, 
                       help="Path to pre-trained scene model checkpoint")
    parser.add_argument("--dataset_path", required=True,
                       help="Path to dataset")
    parser.add_argument("--sequence", required=True,
                       help="Sequence name")
    
    # Style parameters
    parser.add_argument("--style_prompt", type=str, required=True,
                       help="Text prompt for style (e.g., 'a Van Gogh painting')")
    parser.add_argument("--style_image", type=str, default=None,
                       help="Path to style reference image (optional)")
    parser.add_argument("--object_prompt", type=str, default="a photo",
                       help="Object prompt for neutral reference")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=10000,
                       help="Number of training iterations")
    parser.add_argument("--lr_xyz", type=float, default=1.6e-4,
                       help="Learning rate for xyz coordinates")
    parser.add_argument("--lr_feature", type=float, default=2.5e-3,
                       help="Learning rate for features")
    parser.add_argument("--lr_opacity", type=float, default=5.0e-2,
                       help="Learning rate for opacity")
    parser.add_argument("--lr_scaling", type=float, default=5.0e-3,
                       help="Learning rate for scaling")
    parser.add_argument("--lr_rotation", type=float, default=1.0e-3,
                       help="Learning rate for rotation")
    
    # Loss weights
    parser.add_argument("--content_weight", type=float, default=1.0,
                       help="Weight for VGG content loss")
    parser.add_argument("--patch_weight", type=float, default=5000.0,
                       help="Weight for patch CLIP loss")
    parser.add_argument("--directional_weight", type=float, default=2500.0,
                       help="Weight for directional CLIP loss")
    parser.add_argument("--background_weight", type=float, default=100.0,
                       help="Weight for background loss")
    
    # Training settings
    parser.add_argument("--crop_size", type=int, default=224,
                       help="Crop size for patch loss")
    parser.add_argument("--num_crops", type=int, default=4,
                       help="Number of crops for patch loss")
    parser.add_argument("--random_background", action="store_true",
                       help="Use random background colors during training")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--save_iterations", nargs="+", type=int, 
                       default=[5000, 7500, 10000],
                       help="Iterations to save checkpoints")
    parser.add_argument("--preview_interval", type=int, default=500,
                       help="Interval to save preview images")
    
    # Rendering
    parser.add_argument("--render_test", action="store_true",
                       help="Render test sequence after training")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize trainer
    trainer = StyleTransferTrainer(args)
    
    # Run training
    trainer.train()
    
    # Render test sequence if requested
    if args.render_test:
        trainer.render_test_sequence()
    
    print("ML-HUGS style transfer completed successfully!")


if __name__ == "__main__":
    main()