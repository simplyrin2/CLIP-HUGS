# ML-HUGS Style Transfer

This project integrates CLIP Gaussian losses from CLIPGaussian/3D to apply style transfer to scene components in ML-HUGS while keeping human models unchanged.

## Overview

The style transfer system:
1. Loads pre-trained human and scene models from ML-HUGS output
2. Keeps the human model frozen (unchanged)
3. Applies CLIP-based style losses to fine-tune only the scene components
4. Renders the final stylized scenes using the ML-HUGS rendering pipeline

## Files

- `ml_hugs_style_transfer.py` - Main training script for style transfer
- `ml_hugs_style_render.py` - Rendering script for stylized models
- `ml_hugs_style_utils.py` - Utility script for setup and command generation
- `example_usage.sh` - Example usage commands

## Requirements

- PyTorch with CUDA support
- ML-HUGS environment setup
- CLIPGaussian dependencies
- Pre-trained ML-HUGS human and scene models

## Quick Start

### 1. Find Available Models

```bash
python ml_hugs_style_utils.py list-models /path/to/ml-hugs/output
```

### 2. Generate Training Command

```bash
python ml_hugs_style_utils.py gen-train-cmd \
  --human_checkpoint /path/to/human_model.pth \
  --scene_checkpoint /path/to/scene_model.pth \
  --dataset_path /path/to/dataset \
  --sequence sequence_name \
  --style_prompt "a Van Gogh painting" \
  --output_dir ./output/van_gogh_style
```

### 3. Run Style Transfer Training

```bash
python ml_hugs_style_transfer.py \
  --human_checkpoint /path/to/human_model.pth \
  --scene_checkpoint /path/to/scene_model.pth \
  --dataset_path /path/to/dataset \
  --sequence sequence_name \
  --style_prompt "a Van Gogh painting" \
  --output_dir ./output/van_gogh_style \
  --iterations 10000 \
  --render_test
```

### 4. Render Stylized Results

```bash
python ml_hugs_style_render.py \
  --human_checkpoint /path/to/human_model.pth \
  --stylized_scene_checkpoint ./output/van_gogh_style/checkpoints/scene_stylized_final.pth \
  --dataset_path /path/to/dataset \
  --sequence sequence_name \
  --output_dir ./output/van_gogh_renders \
  --render_comparison \
  --create_video
```

## Training Parameters

### Model Paths
- `--human_checkpoint`: Path to pre-trained human model
- `--scene_checkpoint`: Path to pre-trained scene model  
- `--dataset_path`: Path to dataset
- `--sequence`: Sequence name

### Style Parameters
- `--style_prompt`: Text description of desired style (e.g., "a Van Gogh painting")
- `--style_image`: Optional path to style reference image
- `--object_prompt`: Neutral object description (default: "a photo")

### Loss Weights
- `--content_weight`: VGG content loss weight (default: 1.0)
- `--patch_weight`: Patch CLIP loss weight (default: 5000.0)
- `--directional_weight`: Directional CLIP loss weight (default: 2500.0)
- `--background_weight`: Background loss weight (default: 100.0)

### Training Settings
- `--iterations`: Number of training iterations (default: 10000)
- `--lr_xyz`: Learning rate for positions (default: 1.6e-4)
- `--lr_feature`: Learning rate for features (default: 2.5e-3)
- `--lr_opacity`: Learning rate for opacity (default: 5.0e-2)
- `--lr_scaling`: Learning rate for scaling (default: 5.0e-3)
- `--lr_rotation`: Learning rate for rotation (default: 1.0e-3)

## Rendering Options

### Basic Rendering
- `--render_mode`: Choose 'human', 'scene', or 'human_scene'
- `--split`: Dataset split to render ('train', 'test', 'val')

### Special Rendering Modes
- `--render_comparison`: Render all modes for comparison
- `--render_rotation`: Create rotating camera sequence
- `--create_video`: Generate MP4 video from images

## Architecture

### StyleTransferTrainer Class
Main training class that:
- Loads pre-trained human and scene models
- Sets up CLIP and VGG models for style loss computation
- Manages training loop with frozen human model
- Computes multi-component style losses
- Handles checkpointing and visualization

### Key Methods
- `load_human_model()`: Load and freeze human model
- `load_scene_model()`: Load scene model and setup optimizer
- `setup_style_embedding()`: Compute target style embedding
- `compute_style_losses()`: Calculate all style transfer losses
- `train_step()`: Single training iteration

### Loss Components
1. **VGG Content Loss**: Preserves content structure
2. **Patch CLIP Loss**: Style transfer on image patches
3. **Directional CLIP Loss**: Global style direction matching
4. **Background Loss**: Optional background regularization

## Examples

### Van Gogh Style Transfer
```bash
python ml_hugs_style_transfer.py \
  --human_checkpoint ./models/human_final.pth \
  --scene_checkpoint ./models/scene_final.pth \
  --dataset_path ./data/neuman/bike \
  --sequence bike \
  --style_prompt "a Van Gogh painting with swirling brushstrokes" \
  --output_dir ./output/bike_vangogh \
  --iterations 15000 \
  --patch_weight 7500.0 \
  --directional_weight 3000.0
```

### Watercolor Style Transfer  
```bash
python ml_hugs_style_transfer.py \
  --human_checkpoint ./models/human_final.pth \
  --scene_checkpoint ./models/scene_final.pth \
  --dataset_path ./data/neuman/lab \
  --sequence lab \
  --style_prompt "a watercolor painting with soft colors" \
  --output_dir ./output/lab_watercolor \
  --content_weight 2.0 \
  --patch_weight 4000.0 \
  --directional_weight 2000.0
```

### Using Style Reference Image
```bash
python ml_hugs_style_transfer.py \
  --human_checkpoint ./models/human_final.pth \
  --scene_checkpoint ./models/scene_final.pth \
  --dataset_path ./data/neuman/seattle \
  --sequence seattle \
  --style_image ./styles/monet_waterlilies.jpg \
  --object_prompt "a photograph" \
  --output_dir ./output/seattle_monet
```

## Output Structure

```
output/
├── experiment_config.json          # Experiment configuration
├── checkpoints/                    # Model checkpoints
│   ├── scene_stylized_005000.pth
│   ├── scene_stylized_010000.pth
│   └── scene_stylized_final.pth
├── previews/                       # Training preview images
│   ├── rendered_000500_001.png
│   └── gt_001.png
├── test_renders/                   # Test sequence renders
└── events.out.tfevents.*          # Tensorboard logs
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or image resolution
   - Use gradient checkpointing
   - Process fewer crops for patch loss

2. **Model Loading Errors**
   - Verify checkpoint paths and formats
   - Check ML-HUGS model compatibility
   - Ensure proper CUDA device setup

3. **Style Transfer Quality**
   - Adjust loss weights for better balance
   - Try different style prompts or reference images
   - Increase training iterations
   - Experiment with learning rates

### Performance Tips
- Use tensorboard to monitor training progress
- Save checkpoints frequently for long training runs
- Render preview images to evaluate style transfer quality
- Use validation views to check overfitting

## Integration with ML-HUGS

This system is designed to work with standard ML-HUGS outputs:
- Human models (`.pth` files) remain unchanged
- Scene models (`.pth` files) are fine-tuned for style
- Original ML-HUGS rendering pipeline is preserved
- Compatible with existing ML-HUGS datasets and configurations

The style transfer specifically targets scene Gaussians while preserving human geometry and appearance, allowing for realistic human-scene style combinations.