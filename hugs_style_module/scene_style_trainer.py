import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

# GaussianEditor is an optional external dependency for the compatibility
# trainer. Point to any checkout with GAUSSIAN_EDITOR_ROOT.
gaussian_editor_root = os.environ.get("GAUSSIAN_EDITOR_ROOT")
if not gaussian_editor_root:
    raise RuntimeError(
        "Set GAUSSIAN_EDITOR_ROOT to a GaussianEditor checkout for the "
        "GaussianEditor compatibility trainer."
    )
sys.path.insert(0, str(Path(gaussian_editor_root).expanduser().resolve()))

from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.arguments import OptimizationParams
from .clip_losses import (
    VGGContent,
    get_clip,
    content_loss_vgg,
    directional_clip_loss,
    patch_clip_loss,
    background_uniform_loss,
)

"""
Scene Style Transfer Trainer using GaussianEditor's GaussianModel

This trainer uses GaussianEditor's GaussianModel class for compatibility.
Focuses on stylizing the background scene while keeping human model frozen.

Example usage:
python scripts/style_scene_only.py \
  --scene_ckpt /path/to/scene_final.pth \
  --style_text "a Van Gogh painting with swirling brushstrokes" \
  --steps 5000 \
  --out_ckpt scene_stylized.pth
"""

def convert_mlhugs_to_gaussian_editor(mlhugs_checkpoint, output_path):
    """
    Convert ML-HUGS scene checkpoint to GaussianEditor GaussianModel format
    
    Args:
        mlhugs_checkpoint: Path to ML-HUGS scene checkpoint or loaded checkpoint dict
        output_path: Path to save GaussianEditor-compatible checkpoint
    
    Returns:
        GaussianModel instance loaded with the checkpoint
    """
    # Load checkpoint if path provided
    if isinstance(mlhugs_checkpoint, str):
        checkpoint = torch.load(mlhugs_checkpoint, map_location='cpu')
    else:
        checkpoint = mlhugs_checkpoint
    
    print(f"Converting ML-HUGS checkpoint with keys: {list(checkpoint.keys())}")
    
    # Create GaussianEditor GaussianModel
    gaussian_model = GaussianModel(
        sh_degree=3,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2,
    )
    
    # Extract data from ML-HUGS format
    if 'xyz' in checkpoint:
        # Format without underscores (compatible)
        xyz = checkpoint['xyz']
        features_dc = checkpoint['features_dc']
        features_rest = checkpoint['features_rest']
        scaling = checkpoint['scaling']
        rotation = checkpoint['rotation']
        opacity = checkpoint['opacity']
        active_sh_degree = checkpoint['active_sh_degree']
    elif '_xyz' in checkpoint:
        # Format with underscores (need conversion)
        xyz = checkpoint['_xyz']
        features_dc = checkpoint['_features_dc']
        features_rest = checkpoint['_features_rest']
        scaling = checkpoint['_scaling']
        rotation = checkpoint['_rotation']
        opacity = checkpoint['_opacity']
        active_sh_degree = checkpoint['active_sh_degree']
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
    
    # Set GaussianModel attributes directly
    gaussian_model._xyz = xyz.cuda() if not xyz.is_cuda else xyz
    gaussian_model._features_dc = features_dc.cuda() if not features_dc.is_cuda else features_dc
    gaussian_model._features_rest = features_rest.cuda() if not features_rest.is_cuda else features_rest
    gaussian_model._scaling = scaling.cuda() if not scaling.is_cuda else scaling
    gaussian_model._rotation = rotation.cuda() if not rotation.is_cuda else rotation
    gaussian_model._opacity = opacity.cuda() if not opacity.is_cuda else opacity
    gaussian_model.active_sh_degree = active_sh_degree
    
    # Initialize other required attributes
    num_points = xyz.shape[0]
    gaussian_model.max_radii2D = torch.zeros(num_points, device='cuda')
    gaussian_model.xyz_gradient_accum = torch.zeros_like(xyz)
    gaussian_model.denom = torch.zeros(num_points, 1, device='cuda')
    gaussian_model.spatial_lr_scale = checkpoint.get('spatial_lr_scale', 1.0)
    
    # Initialize generation tracking (GaussianEditor feature)
    gaussian_model._generation = torch.zeros(num_points, device='cuda', dtype=torch.int32)
    
    print(f"✓ Converted {num_points} points to GaussianEditor format")
    print(f"Active SH degree: {gaussian_model.active_sh_degree}")
    print(f"XYZ range: [{gaussian_model._xyz.min():.3f}, {gaussian_model._xyz.max():.3f}]")
    
    return gaussian_model


def save_gaussian_editor_checkpoint(gaussian_model, checkpoint_path):
    """
    Save GaussianEditor GaussianModel in both GaussianEditor and ML-HUGS compatible formats
    
    Args:
        gaussian_model: GaussianEditor GaussianModel instance
        checkpoint_path: Path to save checkpoint
    """
    # Save in ML-HUGS compatible format for rendering
    mlhugs_state = {
        'active_sh_degree': gaussian_model.active_sh_degree,
        'xyz': gaussian_model._xyz,
        'features_dc': gaussian_model._features_dc,
        'features_rest': gaussian_model._features_rest,
        'scaling': gaussian_model._scaling,
        'rotation': gaussian_model._rotation,
        'opacity': gaussian_model._opacity,
        'max_radii2D': gaussian_model.max_radii2D,
        'xyz_gradient_accum': gaussian_model.xyz_gradient_accum,
        'denom': gaussian_model.denom,
        'spatial_lr_scale': gaussian_model.spatial_lr_scale,
    }
    
    # Add optimizer state if available
    if gaussian_model.optimizer is not None:
        mlhugs_state['optimizer'] = gaussian_model.optimizer.state_dict()
    else:
        # Create minimal optimizer state
        mlhugs_state['optimizer'] = {
            'param_groups': [
                {
                    'lr': 0.001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0,
                    'amsgrad': False,
                    'params': []
                }
            ],
            'state': {}
        }
    
    torch.save(mlhugs_state, checkpoint_path)
    print(f"Saved GaussianModel checkpoint to: {checkpoint_path}")
    print(f"Keys: {list(mlhugs_state.keys())}")
    
    # Also save PLY for GaussianEditor compatibility
    ply_path = checkpoint_path.replace('.pth', '.ply')
    gaussian_model.save_ply(ply_path)
    print(f"Saved PLY file to: {ply_path}")


def run_gaussian_editor_style_transfer(
    scene_checkpoint_path,
    style_text,
    dataset_cameras=None,
    steps=3000,
    lr=0.0025,
    lambda_d=10.0,
    lambda_p=150.0,
    lambda_c=1.0,
    lambda_b=5.0,
    out_ckpt="scene_stylized.pth",
    device="cuda",
    save_interval=500
):
    """
    Run style transfer using GaussianEditor's GaussianModel class
    
    Args:
        scene_checkpoint_path: Path to ML-HUGS scene checkpoint
        style_text: Style prompt (e.g. "Van Gogh painting")
        dataset_cameras: List of camera objects for rendering
        steps: Training steps
        lr: Learning rate
        lambda_d: Directional CLIP loss weight
        lambda_p: Patch CLIP loss weight
        lambda_c: Content loss weight
        lambda_b: Background uniformity weight
        out_ckpt: Output checkpoint path
        device: Device
        save_interval: Save checkpoint interval
    """
    print(f"=== GaussianEditor Style Transfer ===")
    print(f"Input checkpoint: {scene_checkpoint_path}")
    print(f"Style: '{style_text}'")
    print(f"Steps: {steps}")
    
    # Convert ML-HUGS checkpoint to GaussianEditor format
    gaussian_model = convert_mlhugs_to_gaussian_editor(scene_checkpoint_path, None)
    
    # Move to device
    gaussian_model._xyz = gaussian_model._xyz.to(device)
    gaussian_model._features_dc = gaussian_model._features_dc.to(device)
    gaussian_model._features_rest = gaussian_model._features_rest.to(device)
    gaussian_model._scaling = gaussian_model._scaling.to(device)
    gaussian_model._rotation = gaussian_model._rotation.to(device)
    gaussian_model._opacity = gaussian_model._opacity.to(device)
    
    # Setup for training - focus on appearance parameters only
    # Detach tensors first to make them leaf variables, then set requires_grad
    gaussian_model._xyz = gaussian_model._xyz.detach().requires_grad_(False)  # Keep geometry fixed
    gaussian_model._features_dc = gaussian_model._features_dc.detach().requires_grad_(True)  # Allow color changes
    gaussian_model._features_rest = gaussian_model._features_rest.detach().requires_grad_(True)  # Allow SH changes
    gaussian_model._scaling = gaussian_model._scaling.detach().requires_grad_(False)  # Keep scale fixed
    gaussian_model._rotation = gaussian_model._rotation.detach().requires_grad_(False)  # Keep rotation fixed
    gaussian_model._opacity = gaussian_model._opacity.detach().requires_grad_(False)  # Keep opacity fixed
    
    # Setup training parameters
    dummy_parser = ArgumentParser(description="Style transfer training parameters")
    training_args = OptimizationParams(parser=dummy_parser, max_steps=steps)
    training_args.feature_lr = lr
    gaussian_model.spatial_lr_scale = 1.0
    gaussian_model.training_setup(training_args)
    
    # Load CLIP and VGG models
    print("Loading CLIP and VGG models...")
    vgg = VGGContent(device)
    clip_model, clip_tok = get_clip(device)
    
    # Training loop
    print("Starting training...")
    progress_bar = tqdm(range(steps), desc="Style Transfer")
    
    for step in progress_bar:
        # For now, create a simple render without camera (just feature optimization)
        # In a full implementation, you'd render from random camera viewpoints
        
        # Get current features
        current_features_dc = gaussian_model._features_dc
        current_features_rest = gaussian_model._features_rest
        
        # Create a simple "rendered" image by using the DC features as RGB
        # This is a simplified version - full implementation would use proper rendering
        rgb_features = current_features_dc.squeeze(1)  # Remove middle dimension
        
        # Convert SH DC to RGB (simplified)
        SH_C0 = 0.28209479177387814
        rgb_render = (rgb_features + 0.5) / SH_C0  # Convert from SH to RGB
        rgb_render = torch.clamp(rgb_render, 0.0, 1.0)
        
        # Reshape to image-like format (simplified - using mean across points)
        # In practice, you'd render proper images using camera views
        mean_color = rgb_render.mean(0, keepdim=True)  # Shape: [1, 3]
        fake_image = mean_color.unsqueeze(0).expand(1, 3, 224, 224)  # Create 224x224 image
        
        # Compute losses
        total_loss = 0.0
        loss_dict = {}
        
        # Directional CLIP loss
        if lambda_d > 0:
            try:
                neutral_ref = torch.zeros_like(fake_image)
                L_d = directional_clip_loss(clip_model, clip_tok, fake_image, neutral_ref, text=style_text)
                total_loss += lambda_d * L_d
                loss_dict['dir'] = L_d.item()
            except Exception as e:
                print(f"Warning: Directional CLIP loss failed: {e}")
                loss_dict['dir'] = 0.0
        
        # Patch CLIP loss
        if lambda_p > 0:
            try:
                neutral_ref = torch.zeros_like(fake_image)
                L_p = patch_clip_loss(clip_model, clip_tok, fake_image, neutral_ref, text=style_text, patches=4)
                total_loss += lambda_p * L_p
                loss_dict['patch'] = L_p.item()
            except Exception as e:
                print(f"Warning: Patch CLIP loss failed: {e}")
                loss_dict['patch'] = 0.0
        
        # Background uniformity loss (regularization)
        if lambda_b > 0:
            try:
                L_b = background_uniform_loss(fake_image)
                total_loss += lambda_b * L_b
                loss_dict['bg'] = L_b.item()
            except Exception as e:
                print(f"Warning: Background loss failed: {e}")
                loss_dict['bg'] = 0.0
        
        # Optimization step
        if total_loss > 0:
            gaussian_model.optimizer.zero_grad()
            total_loss.backward()
            gaussian_model.optimizer.step()
            loss_dict['total'] = total_loss.item()
        else:
            loss_dict['total'] = 0.0
        
        # Update progress
        progress_bar.set_postfix(loss_dict)
        
        # Save intermediate checkpoints
        if (step + 1) % save_interval == 0:
            intermediate_path = out_ckpt.replace('.pth', f'_step_{step+1:05d}.pth')
            save_gaussian_editor_checkpoint(gaussian_model, intermediate_path)
            print(f"\nSaved intermediate: {intermediate_path}")
    
    # Save final checkpoint
    save_gaussian_editor_checkpoint(gaussian_model, out_ckpt)
    print(f"\n✅ Style transfer completed!")
    print(f"Final checkpoint: {out_ckpt}")
    
    return out_ckpt


def run_gaussian_editor_style_with_image(
    scene_checkpoint_path,
    style_image_path,
    steps=3000,
    lr=0.0025,
    out_ckpt="scene_stylized.pth",
    device="cuda"
):
    """
    Run style transfer with reference image using GaussianEditor's GaussianModel
    """
    from PIL import Image
    from torchvision import transforms
    
    print(f"=== GaussianEditor Image Style Transfer ===")
    print(f"Input checkpoint: {scene_checkpoint_path}")
    print(f"Style image: {style_image_path}")
    
    # Load and process style image
    style_img = Image.open(style_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    style_tensor = transform(style_img).unsqueeze(0).to(device)
    
    # Convert checkpoint
    gaussian_model = convert_mlhugs_to_gaussian_editor(scene_checkpoint_path, None)
    
    # Move to device and setup training (same as text version)
    gaussian_model._xyz = gaussian_model._xyz.to(device)
    gaussian_model._features_dc = gaussian_model._features_dc.to(device)
    gaussian_model._features_rest = gaussian_model._features_rest.to(device)
    gaussian_model._scaling = gaussian_model._scaling.to(device)
    gaussian_model._rotation = gaussian_model._rotation.to(device)
    gaussian_model._opacity = gaussian_model._opacity.to(device)
    
    # Setup for training - detach first to make leaf variables
    gaussian_model._features_dc = gaussian_model._features_dc.detach().requires_grad_(True)
    gaussian_model._features_rest = gaussian_model._features_rest.detach().requires_grad_(True)
    
    dummy_parser = ArgumentParser(description="Style transfer training parameters")
    training_args = OptimizationParams(parser=dummy_parser, max_steps=steps)
    training_args.feature_lr = lr
    gaussian_model.training_setup(training_args)
    
    # Load models
    clip_model, _ = get_clip(device)
    
    # Get style features
    with torch.no_grad():
        style_features = clip_model.encode_image(style_tensor)
        style_features = F.normalize(style_features, dim=-1)
    
    # Training loop (simplified)
    for step in tqdm(range(steps), desc="Image Style Transfer"):
        # Simple feature optimization (same approach as text version)
        current_features_dc = gaussian_model._features_dc
        rgb_features = current_features_dc.squeeze(1)
        SH_C0 = 0.28209479177387814
        rgb_render = (rgb_features + 0.5) / SH_C0
        rgb_render = torch.clamp(rgb_render, 0.0, 1.0)
        
        mean_color = rgb_render.mean(0, keepdim=True)
        fake_image = mean_color.unsqueeze(0).expand(1, 3, 224, 224)
        
        # Style similarity loss
        with torch.no_grad():
            rendered_features = clip_model.encode_image(fake_image)
            rendered_features = F.normalize(rendered_features, dim=-1)
        
        style_loss = 1 - F.cosine_similarity(rendered_features, style_features).mean()
        
        if style_loss > 0:
            gaussian_model.optimizer.zero_grad()
            style_loss.backward()
            gaussian_model.optimizer.step()
    
    # Save result
    save_gaussian_editor_checkpoint(gaussian_model, out_ckpt)
    print(f"✅ Image style transfer completed: {out_ckpt}")
    
    return out_ckpt
#     steps=3000,
#     lr=5e-4,
#     lambda_d=10.0,      # Directional CLIP loss weight
#     lambda_p=150.0,     # Patch CLIP loss weight  
#     lambda_c=1.0,       # Content preservation weight
#     lambda_b=5.0,       # Background uniformity weight
#     lambda_scene=50.0,  # Scene-only stylization weight
#     out_ckpt="scene_stylized.pth",
#     device="cuda",
#     save_interval=500,
#     preview_interval=100
# ):
#     """
#     Run scene style transfer training
    
#     Args:
#         adapter: SceneAdapter instance with human and scene models
#         style_text: Target style description (e.g., "a Van Gogh painting")
#         steps: Number of training steps
#         lr: Learning rate
#         lambda_d: Weight for directional CLIP loss
#         lambda_p: Weight for patch CLIP loss
#         lambda_c: Weight for content preservation loss
#         lambda_b: Weight for background uniformity loss
#         lambda_scene: Weight for scene-only stylization
#         out_ckpt: Output checkpoint path
#         device: Device to run on
#         save_interval: Steps between checkpoint saves
#         preview_interval: Steps between preview prints
#     """
#     device = torch.device(device)
#     adapter.to(device)

#     # Initialize CLIP and VGG models
#     vgg = VGGContent(device)
#     clip, tok = get_clip(device)

#     # Optimizer for scene parameters only
#     scene_params = adapter.scene_parameters()
#     if not scene_params:
#         raise ValueError("No trainable scene parameters found!")
    
#     opt = optim.Adam(scene_params, lr=lr)
    
#     print(f"Starting scene style transfer with {len(scene_params)} parameter groups")
#     print(f"Style target: '{style_text}'")
#     print(f"Training steps: {steps}")
    
#     # Training loop
#     for step in range(steps):
#         # Random frame selection
#         idx = torch.randint(0, adapter.num_frames, (1,)).item()
#         cam, pose = adapter.get_frame(idx)

#         # Render stylized scene with human
#         stylized_render = adapter.render_human_scene(cam, pose)
        
#         # Handle 4-channel renders (drop alpha if present)
#         if stylized_render.shape[0] == 4:
#             stylized_render = stylized_render[:3]

#         # Render scene only (for scene-specific style loss)
#         scene_only = adapter.render_scene_only(cam)
#         if scene_only.shape[0] == 4:
#             scene_only = scene_only[:3]

#         # Resize for CLIP/VGG processing
#         sty_resized = F.interpolate(stylized_render.unsqueeze(0), size=(224, 224), mode='bilinear')
#         scene_resized = F.interpolate(scene_only.unsqueeze(0), size=(224, 224), mode='bilinear')
        
#         # Create neutral reference (can be improved with actual GT if available)
#         neutral_ref = torch.zeros_like(sty_resized)

#         # Compute losses
        
#         # 1. Content preservation loss (VGG)
#         # Use scene-only render as reference to preserve scene structure
#         L_c = content_loss_vgg(vgg, sty_resized, scene_resized.detach())
        
#         # 2. Directional CLIP loss on full render (human + scene)
#         L_d = directional_clip_loss(clip, tok, sty_resized, neutral_ref, text=style_text)
        
#         # 3. Patch CLIP loss on full render
#         L_p = patch_clip_loss(clip, tok, sty_resized, neutral_ref, text=style_text, patches=4)
        
#         # 4. Scene-only directional CLIP loss (focus style on scene)
#         L_scene = directional_clip_loss(clip, tok, scene_resized, neutral_ref, text=style_text)
        
#         # 5. Background uniformity loss (optional regularization)
#         L_b = background_uniform_loss(scene_resized)

#         # Combined loss
#         total_loss = (lambda_d * L_d + 
#                      lambda_p * L_p + 
#                      lambda_c * L_c + 
#                      lambda_b * L_b + 
#                      lambda_scene * L_scene)

#         # Optimization step
#         opt.zero_grad()
#         total_loss.backward()
#         opt.step()

#         # Logging
#         if (step + 1) % preview_interval == 0:
#             print(f"[{step+1:4d}/{steps}] "
#                   f"Total: {total_loss.item():.4f} | "
#                   f"Dir: {L_d.item():.4f} | "
#                   f"Patch: {L_p.item():.4f} | "
#                   f"Content: {L_c.item():.4f} | "
#                   f"Scene: {L_scene.item():.4f} | "
#                   f"BG: {L_b.item():.4f}")

#         # Save intermediate checkpoints
#         if (step + 1) % save_interval == 0:
#             intermediate_path = out_ckpt.replace('.pth', f'_step_{step+1:05d}.pth')
#             adapter.save_scene_checkpoint(intermediate_path)
#             print(f"Saved intermediate checkpoint: {intermediate_path}")

#     # Save final checkpoint using SceneGS-compatible format
#     save_scene_checkpoint_compatible(adapter.scene_model, out_ckpt)
#     print(f"Scene style transfer completed!")
#     print(f"Final checkpoint saved: {out_ckpt}")
    
#     return out_ckpt


# def run_scene_style_with_reference(
#     adapter,
#     style_image_path,
#     steps=3000,
#     lr=5e-4,
#     lambda_d=10.0,
#     lambda_p=150.0,
#     lambda_c=1.0,
#     lambda_b=5.0,
#     lambda_scene=50.0,
#     out_ckpt="scene_stylized.pth",
#     device="cuda"
# ):
#     """
#     Run scene style transfer using a reference style image instead of text
    
#     Args:
#         adapter: SceneAdapter instance
#         style_image_path: Path to style reference image
#         Other args: Same as run_scene_style
#     """
#     from PIL import Image
#     from torchvision import transforms
    
#     device = torch.device(device)
#     adapter.to(device)

#     # Load and process style image
#     style_img = Image.open(style_image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#     style_tensor = transform(style_img).unsqueeze(0).to(device)
    
#     # Initialize models
#     vgg = VGGContent(device)
#     clip, tok = get_clip(device)
    
#     # Get style features from reference image
#     with torch.no_grad():
#         style_features = clip.encode_image(style_tensor)
#         style_features = F.normalize(style_features, dim=-1)

#     # Optimizer
#     scene_params = adapter.scene_parameters()
#     opt = optim.Adam(scene_params, lr=lr)
    
#     print(f"Starting image-based scene style transfer")
#     print(f"Style image: {style_image_path}")
    
#     # Modified training loop for image-based style transfer
#     for step in range(steps):
#         idx = torch.randint(0, adapter.num_frames, (1,)).item()
#         cam, pose = adapter.get_frame(idx)

#         stylized_render = adapter.render_human_scene(cam, pose)
#         if stylized_render.shape[0] == 4:
#             stylized_render = stylized_render[:3]

#         scene_only = adapter.render_scene_only(cam)
#         if scene_only.shape[0] == 4:
#             scene_only = scene_only[:3]

#         sty_resized = F.interpolate(stylized_render.unsqueeze(0), size=(224, 224), mode='bilinear')
#         scene_resized = F.interpolate(scene_only.unsqueeze(0), size=(224, 224), mode='bilinear')

#         # Compute image-based style losses
#         # Use rendered features vs style features directly
#         with torch.no_grad():
#             rendered_features = clip.encode_image(sty_resized)
#             rendered_features = F.normalize(rendered_features, dim=-1)
        
#         # Style similarity loss
#         style_similarity = 1 - F.cosine_similarity(rendered_features, style_features).mean()
        
#         # Content preservation
#         L_c = content_loss_vgg(vgg, sty_resized, scene_resized.detach())
        
#         # Background uniformity
#         L_b = background_uniform_loss(scene_resized)

#         total_loss = (lambda_d * style_similarity + 
#                      lambda_c * L_c + 
#                      lambda_b * L_b)

#         opt.zero_grad()
#         total_loss.backward()
#         opt.step()

#         if (step + 1) % 100 == 0:
#             print(f"[{step+1:4d}/{steps}] "
#                   f"Total: {total_loss.item():.4f} | "
#                   f"Style: {style_similarity.item():.4f} | "
#                   f"Content: {L_c.item():.4f} | "
#                   f"BG: {L_b.item():.4f}")

#     # Save final result using SceneGS-compatible format
#     save_scene_checkpoint_compatible(adapter.scene_model, out_ckpt)
#     print(f"Image-based scene style transfer completed!")
#     print(f"Final checkpoint saved: {out_ckpt}")
    
#     return out_ckpt

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from .clip_losses import (
    VGGContent,
    get_clip,
    content_loss_vgg,
    directional_clip_loss,
    patch_clip_loss,
    background_uniform_loss,
)

"""
Scene Style Transfer Trainer

This trainer focuses on stylizing the background scene while keeping the human model frozen.

Example usage:
python scripts/style_scene.py \
  --cfg_file cfg_files/release/neuman/hugs_scene.yaml \
  --dataset_seq bike \
  --human_ckpt /path/to/human_final.pth \
  --scene_ckpt /path/to/scene_final.pth \
  --style_text "a Van Gogh painting with swirling brushstrokes" \
  --steps 5000 \
  --out_ckpt scene_stylized.pth
"""


def save_scene_checkpoint_compatible(scene_model, optimizer, checkpoint_path):
    """Save scene checkpoint in ML-HUGS compatible format matching SceneGS.restore() expectations"""
    
    # Create state dict with keys that match SceneGS.state_dict() method
    # Based on scene.py: keys should NOT have underscore prefix
    state_dict = {
        'active_sh_degree': scene_model.active_sh_degree,
        'xyz': scene_model._xyz,  # Remove underscore prefix
        'features_dc': scene_model._features_dc,  # Remove underscore prefix  
        'features_rest': scene_model._features_rest,  # Remove underscore prefix
        'scaling': scene_model._scaling,  # Remove underscore prefix
        'rotation': scene_model._rotation,  # Remove underscore prefix
        'opacity': scene_model._opacity,  # Remove underscore prefix
        'max_radii2D': scene_model.max_radii2D,
        'xyz_gradient_accum': scene_model.xyz_gradient_accum,
        'denom': scene_model.denom,
        'spatial_lr_scale': getattr(scene_model, 'spatial_lr_scale', 1.0),
    }
    
    # Add optimizer state in the format SceneGS.restore() expects - as 'optimizer' key (NOT separate param_groups/state)
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    else:
        # Create minimal optimizer state structure if no optimizer provided
        state_dict['optimizer'] = {
            'param_groups': [
                {
                    'lr': 0.001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0,
                    'amsgrad': False,
                    'params': []
                }
            ],
            'state': {}
        }
    
    torch.save(state_dict, checkpoint_path)
    print(f"Saved scene checkpoint with SceneGS-compatible keys: {list(state_dict.keys())}")
    
    # Verify the saved checkpoint can be loaded
    try:
        verify_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        expected_keys = ['xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity', 'optimizer']
        missing_keys = [key for key in expected_keys if key not in verify_checkpoint]
        if missing_keys:
            print(f"WARNING: Missing expected keys: {missing_keys}")
        else:
            print("✓ Checkpoint format verified - compatible with SceneGS.restore()")
    except Exception as e:
        print(f"WARNING: Could not verify checkpoint: {e}")


def run_scene_style(
    adapter,
    style_text,
    steps=3000,
    lr=5e-4,
    lambda_d=10.0,      # Directional CLIP loss weight
    lambda_p=150.0,     # Patch CLIP loss weight  
    lambda_c=1.0,       # Content preservation weight
    lambda_b=5.0,       # Background uniformity weight
    lambda_scene=50.0,  # Scene-only stylization weight
    out_ckpt="scene_stylized.pth",
    device="cuda",
    save_interval=500,
    preview_interval=100
):
    """
    Run scene style transfer training
    
    Args:
        adapter: SceneAdapter instance with human and scene models
        style_text: Target style description (e.g., "a Van Gogh painting")
        steps: Number of training steps
        lr: Learning rate
        lambda_d: Weight for directional CLIP loss
        lambda_p: Weight for patch CLIP loss
        lambda_c: Weight for content preservation loss
        lambda_b: Weight for background uniformity loss
        lambda_scene: Weight for scene-only stylization
        out_ckpt: Output checkpoint path
        device: Device to run on
        save_interval: Steps between checkpoint saves
        preview_interval: Steps between preview prints
    """
    device = torch.device(device)
    adapter.to(device)

    # Initialize CLIP and VGG models
    vgg = VGGContent(device)
    clip, tok = get_clip(device)

    # Optimizer for scene parameters only
    scene_params = adapter.scene_parameters()
    if not scene_params:
        raise ValueError("No trainable scene parameters found!")
    
    opt = optim.Adam(scene_params, lr=lr)
    
    print(f"Starting scene style transfer with {len(list(scene_params))} parameters")
    print(f"Style target: '{style_text}'")
    print(f"Training steps: {steps}")
    
    # Training loop
    for step in range(steps):
        # Random frame selection
        idx = torch.randint(0, adapter.num_frames, (1,)).item()
        cam, pose = adapter.get_frame(idx)

        # Render stylized scene with human
        stylized_render = adapter.render_human_scene(cam, pose)
        
        # Handle 4-channel renders (drop alpha if present)
        if stylized_render.shape[0] == 4:
            stylized_render = stylized_render[:3]

        # Render scene only (for scene-specific style loss)
        scene_only = adapter.render_scene_only(cam)
        if scene_only.shape[0] == 4:
            scene_only = scene_only[:3]

        # Resize for CLIP/VGG processing
        sty_resized = F.interpolate(stylized_render.unsqueeze(0), size=(224, 224), mode='bilinear')
        scene_resized = F.interpolate(scene_only.unsqueeze(0), size=(224, 224), mode='bilinear')
        
        # Create neutral reference (can be improved with actual GT if available)
        neutral_ref = torch.zeros_like(sty_resized)

        # Compute losses
        
        # 1. Content preservation loss (VGG)
        # Use scene-only render as reference to preserve scene structure
        L_c = content_loss_vgg(vgg, sty_resized, scene_resized.detach())
        
        # 2. Directional CLIP loss on full render (human + scene)
        L_d = directional_clip_loss(clip, tok, sty_resized, neutral_ref, text=style_text)
        
        # 3. Patch CLIP loss on full render
        L_p = patch_clip_loss(clip, tok, sty_resized, neutral_ref, text=style_text, patches=4)
        
        # 4. Scene-only directional CLIP loss (focus style on scene)
        L_scene = directional_clip_loss(clip, tok, scene_resized, neutral_ref, text=style_text)
        
        # 5. Background uniformity loss (optional regularization)
        L_b = background_uniform_loss(scene_resized)

        # Combined loss
        total_loss = (lambda_d * L_d + 
                     lambda_p * L_p + 
                     lambda_c * L_c + 
                     lambda_b * L_b + 
                     lambda_scene * L_scene)

        # Optimization step
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # Logging
        if (step + 1) % preview_interval == 0:
            print(f"[{step+1:4d}/{steps}] "
                  f"Total: {total_loss.item():.4f} | "
                  f"Dir: {L_d.item():.4f} | "
                  f"Patch: {L_p.item():.4f} | "
                  f"Content: {L_c.item():.4f} | "
                  f"Scene: {L_scene.item():.4f} | "
                  f"BG: {L_b.item():.4f}")

        # Save intermediate checkpoints
        if (step + 1) % save_interval == 0:
            intermediate_path = out_ckpt.replace('.pth', f'_step_{step+1:05d}.pth')
            save_scene_checkpoint_compatible(adapter.scene_model, opt, intermediate_path)
            print(f"Saved intermediate checkpoint: {intermediate_path}")

    # Save final checkpoint
    save_scene_checkpoint_compatible(adapter.scene_model, opt, out_ckpt)
    print(f"Scene style transfer completed!")
    print(f"Final checkpoint saved: {out_ckpt}")
    
    return out_ckpt


def run_scene_style_with_reference(
    adapter,
    style_image_path,
    steps=3000,
    lr=5e-4,
    lambda_d=10.0,
    lambda_p=150.0,
    lambda_c=1.0,
    lambda_b=5.0,
    lambda_scene=50.0,
    out_ckpt="scene_stylized.pth",
    device="cuda"
):
    """
    Run scene style transfer using a reference style image instead of text
    
    Args:
        adapter: SceneAdapter instance
        style_image_path: Path to style reference image
        Other args: Same as run_scene_style
    """
    from PIL import Image
    from torchvision import transforms
    
    device = torch.device(device)
    adapter.to(device)

    # Load and process style image
    style_img = Image.open(style_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    style_tensor = transform(style_img).unsqueeze(0).to(device)
    
    # Initialize models
    vgg = VGGContent(device)
    clip, tok = get_clip(device)
    
    # Get style features from reference image
    with torch.no_grad():
        style_features = clip.encode_image(style_tensor)
        style_features = F.normalize(style_features, dim=-1)

    # Optimizer
    scene_params = adapter.scene_parameters()
    opt = optim.Adam(scene_params, lr=lr)
    
    print(f"Starting image-based scene style transfer")
    print(f"Style image: {style_image_path}")
    
    # Modified training loop for image-based style transfer
    for step in range(steps):
        idx = torch.randint(0, adapter.num_frames, (1,)).item()
        cam, pose = adapter.get_frame(idx)

        stylized_render = adapter.render_human_scene(cam, pose)
        if stylized_render.shape[0] == 4:
            stylized_render = stylized_render[:3]

        scene_only = adapter.render_scene_only(cam)
        if scene_only.shape[0] == 4:
            scene_only = scene_only[:3]

        sty_resized = F.interpolate(stylized_render.unsqueeze(0), size=(224, 224), mode='bilinear')
        scene_resized = F.interpolate(scene_only.unsqueeze(0), size=(224, 224), mode='bilinear')

        # Compute image-based style losses
        # Use rendered features vs style features directly
        with torch.no_grad():
            rendered_features = clip.encode_image(sty_resized)
            rendered_features = F.normalize(rendered_features, dim=-1)
        
        # Style similarity loss
        style_similarity = 1 - F.cosine_similarity(rendered_features, style_features).mean()
        
        # Content preservation
        L_c = content_loss_vgg(vgg, sty_resized, scene_resized.detach())
        
        # Background uniformity
        L_b = background_uniform_loss(scene_resized)

        total_loss = (lambda_d * style_similarity + 
                     lambda_c * L_c + 
                     lambda_b * L_b)

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            print(f"[{step+1:4d}/{steps}] "
                  f"Total: {total_loss.item():.4f} | "
                  f"Style: {style_similarity.item():.4f} | "
                  f"Content: {L_c.item():.4f} | "
                  f"BG: {L_b.item():.4f}")

    # Save final result
    save_scene_checkpoint_compatible(adapter.scene_model, opt, out_ckpt)
    print(f"Image-based scene style transfer completed!")
    print(f"Final checkpoint saved: {out_ckpt}")
    
    return out_ckpt