# CLIP-HUGS

**CLIP-HUGS** integrates CLIPGaussian-style Gaussian-splatting style transfer
with [ML-HUGS](https://github.com/apple/ml-hugs). It enables text- or
image-guided stylization of ML-HUGS human and scene representations while
preserving the original differentiable rendering workflow.

> This repository is an integration layer. ML-HUGS datasets, SMPL assets,
> pretrained checkpoints, CUDA extensions, and the full runtime checkout are
> intentionally kept external.

## Highlights

- Scene-only stylization while keeping the reconstructed human unchanged.
- Joint human-and-scene style-transfer workflows.
- CLIP directional and patch losses for semantic and local style alignment.
- VGG perceptual content loss for structure preservation.
- ML-HUGS-native adapters, trainers, renderers, and evaluation scripts.
- Optional PLY-based training through the CLIPGaussian 3D implementation.

## Repository layout

```text
CLIP-HUGS/
├── pipeline/              # Unified ML-HUGS style-transfer and rendering pipeline
├── hugs_style_module/     # Style losses, adapters, and ML-HUGS trainers
├── hugs_style_scripts/    # Stylization, rendering, and evaluation entry points
├── clipgaussian_core/     # Curated CLIPGaussian 3D implementation
├── configs/               # ML-HUGS Neuman configuration files
├── dependencies/          # Dependency manifests
├── styles/                # Example style reference images
└── workflows/             # Example shell workflows
```

## How it works

CLIP-HUGS supports two related workflows:

### Unified ML-HUGS pipeline

The scripts in `pipeline/` and `hugs_style_module/` load ML-HUGS human and
scene models directly. The human representation can be frozen while scene
Gaussian parameters are optimized using CLIP and VGG losses. The resulting
model can be rendered with the ML-HUGS renderer.

### CLIPGaussian PLY pipeline

The code in `clipgaussian_core/` provides a curated 3D CLIPGaussian path for
ML-HUGS-exported PLY splats. `clipgaussian_core/run_with_hug.sh` demonstrates
how to pass ML-HUGS-compatible PLY data to the CLIPGaussian style trainer.

## Requirements

- Linux with a CUDA-capable GPU
- Conda
- The existing `ml_hugs` environment
- An ML-HUGS checkout and its compatible datasets/checkpoints
- CLIP and PyTorch dependencies listed in
  [`dependencies/`](dependencies/)

The examples below use the environment already configured for this project:

```bash
conda activate ml_hugs
```

For non-interactive shells, use the active environment's interpreter directly:

```bash
python
```

## Setup

Clone or download ML-HUGS separately, then expose it to Python:

```bash
export ML_HUGS_ROOT=/path/to/ml-hugs
export CLIPGAUSSIAN_ROOT=/path/to/CLIPGaussian/3D
export GAUSSIAN_EDITOR_ROOT=/path/to/GaussianEditor
export PYTHONPATH="$ML_HUGS_ROOT:$PWD:$PYTHONPATH"
```

The ML-HUGS checkout must provide its normal `hugs/` package, datasets,
renderer dependencies, and model assets. The configuration files in
[`configs/`](configs/) correspond to ML-HUGS's
`cfg_files/release/neuman/` configurations.

The optional PLY pipeline additionally requires `CLIPGAUSSIAN_ROOT` to point
to a CLIPGaussian 3D checkout containing `models/gs/` and `arguments/`.
The GaussianEditor compatibility trainer additionally requires
`GAUSSIAN_EDITOR_ROOT` to point to a GaussianEditor checkout.
Open3D is optional and is listed separately in
[`dependencies/optional-visualization.txt`](dependencies/optional-visualization.txt)
because wheels vary by Python and platform.
The libigl binding is similarly separated in
[`dependencies/optional-geometry.txt`](dependencies/optional-geometry.txt);
install it from conda-forge if the selected dataset path uses those geometry
utilities.

ML-HUGS's CUDA extensions must be built for the active environment. After
activating `ml_hugs`, install the extensions from the ML-HUGS checkout:

```bash
python -m pip install --no-build-isolation \
  -e "$ML_HUGS_ROOT/submodules/simple-knn" \
  -e "$ML_HUGS_ROOT/submodules/diff-gaussian-rasterization"
```

The extension build requires a compatible CUDA toolkit and compiler.

Install any missing packages using the manifests in
[`dependencies/`](dependencies/). Do not install a second environment unless
the existing `ml_hugs` environment is incompatible with your ML-HUGS checkout.
The included requirements use Python-version markers for packages whose
available wheels differ between Python versions.

## Quick start

### Unified scene-and-human pipeline

```bash
python pipeline/ml_hugs_style_transfer.py \
  --human_checkpoint /path/to/human_final.pth \
  --scene_checkpoint /path/to/scene_final.pth \
  --dataset_path /path/to/dataset \
  --sequence lab \
  --style_prompt "Starry Night by Vincent van Gogh" \
  --output_dir ./output_style/lab_starry \
  --iterations 8000
```

### Scene-only stylization

Use this workflow when the human should remain unchanged or when only a scene
checkpoint is available:

```bash
python hugs_style_scripts/style_scene_only.py \
  --cfg_file configs/hugs_scene.yaml \
  --dataset_seq lab \
  --scene_ckpt /path/to/scene_final.pth \
  --style_text "a soft watercolor painting" \
  --steps 3000 \
  --out_ckpt ./output_style/scene_watercolor.pth
```

### Render a stylized scene

```bash
python pipeline/ml_hugs_style_render.py \
  --human_checkpoint /path/to/human_final.pth \
  --stylized_scene_checkpoint ./output_style/lab_starry/checkpoints/scene_stylized_final.pth \
  --dataset_path /path/to/dataset \
  --sequence lab \
  --output_dir ./output_style/lab_starry/renders \
  --render_comparison \
  --create_video
```

### Discover available checkpoints

```bash
python pipeline/ml_hugs_style_utils.py \
  list-models /path/to/ml-hugs/output
```

Additional examples are available in [`workflows/`](workflows/).

## Style controls

The unified pipeline supports:

- Text prompts such as `"a Van Gogh painting with swirling brushstrokes"`
- Reference style images through `--style_image`
- CLIP directional and patch-loss weights
- VGG content-preservation weight
- Independent learning rates for Gaussian position, feature, opacity,
  scaling, and rotation parameters
- Training previews, checkpoints, comparison renders, and optional videos

See [`pipeline/README_STYLE_TRANSFER.md`](pipeline/README_STYLE_TRANSFER.md)
for the complete parameter reference.

## Outputs and data policy

Generated checkpoints, renders, logs, datasets, SMPL files, virtual
environments, PyTorch3D, and full Gaussian-splatting source trees are not
included in this repository. Supply them through local paths when running an
experiment. The repository `.gitignore` excludes common generated artifacts
and model files.

## Acknowledgements

This project builds on:

- [ML-HUGS](https://github.com/apple/ml-hugs) for human-scene Gaussian
  representations and rendering.
- [CLIPGaussian](https://github.com/kornelhowil/CLIPGaussian) for
  CLIP-guided Gaussian-splatting style transfer.
- [OpenAI CLIP](https://github.com/openai/CLIP) for image-text embeddings.

Please follow the licenses and citation requirements of each upstream
project. The curated files in this repository retain their upstream
copyright and license notices where applicable.

## Citation

If you use this integration in academic work, please cite the upstream
projects and the CLIPGaussian paper. Add the final CLIP-HUGS citation here
once this integration has a citable publication or project record.

## Status

This repository is an experimental integration and may require small path or
dependency adjustments for a particular ML-HUGS checkout, CUDA version, or
dataset configuration.
