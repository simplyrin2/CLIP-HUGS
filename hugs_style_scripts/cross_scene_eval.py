import os
import sys
import json
import glob
import torch
import argparse
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))
from runtime import configure_runtime

configure_runtime()


'''
# Human from one sequence, scene from another:
# python scripts/cross_scene_eval.py --human_model_dir /path/to/human \
#   --scene_model_dir /path/to/scene --output_dir outputs/cross_scene
# Human from lab, scene from seattle
python scripts/render_cross_scene.py \
  --human_model_dir /path/to/pretrained/lab \
  --scene_model_dir /path/to/pretrained/seattle \
  --output_dir /path/to/output/lab_human_seattle_scene
'''

from hugs.trainer import GaussianTrainer
from hugs.utils.general import safe_state
from hugs.cfg.config import cfg as default_cfg

def get_logger(cfg):
    logdir = cfg.logdir
    mode = 'render_cross_scene'
    logger.add(os.path.join(cfg.logdir, f'{mode}.log'), level='INFO')
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_model_dir", required=True, help="path to trained human model directory (e.g., bike)")
    parser.add_argument("--scene_model_dir", required=True, help="path to trained scene model directory (e.g., lab)")
    parser.add_argument("--output_dir", required=True, help="path to output directory")
    
    args, extras = parser.parse_known_args()
    
    # Load scene config as base (for camera poses, dataset structure)
    scene_cfg_file = os.path.join(args.scene_model_dir, 'config_train.yaml')
    scene_cfg = OmegaConf.load(scene_cfg_file)
    
    # Merge with defaults and CLI args
    cfg = OmegaConf.merge(default_cfg, scene_cfg, OmegaConf.from_cli(extras))
    cfg.eval = True
    cfg.mode = 'human_scene'  # Render both human and scene
    cfg.logdir = args.output_dir
    cfg.logdir_ckpt = args.output_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    get_logger(cfg)
    
    # Find human checkpoint from human model directory
    human_ckpt_files = glob.glob(args.human_model_dir + '/*human*.pth')
    human_ckpt_files += glob.glob(args.human_model_dir + '/ckpt/*human*.pth')
    human_ckpt_files = sorted(human_ckpt_files)
    
    if len(human_ckpt_files) > 0:
        latest_human_ckpt = human_ckpt_files[-1]
        logger.info(f'Using human ckpt from {args.human_model_dir}: {latest_human_ckpt}')
        cfg.human.ckpt = latest_human_ckpt
    else:
        logger.error(f'No human ckpt found in {args.human_model_dir}')
        exit()
    
    # Find scene checkpoint from scene model directory
    scene_ckpt_files = glob.glob(args.scene_model_dir + '/*scene*.pth')
    scene_ckpt_files += glob.glob(args.scene_model_dir + '/ckpt/*scene*.pth')
    scene_ckpt_files = sorted(scene_ckpt_files)
    
    if len(scene_ckpt_files) > 0:
        latest_scene_ckpt = scene_ckpt_files[-1]
        logger.info(f'Using scene ckpt from {args.scene_model_dir}: {latest_scene_ckpt}')
        cfg.scene.ckpt = latest_scene_ckpt
    else:
        logger.error(f'No scene ckpt found in {args.scene_model_dir}')
        exit()
    
    # Initialize trainer
    safe_state(seed=cfg.seed)
    trainer = GaussianTrainer(cfg)
    
    # Run validation and animation
    trainer.validate()
    trainer.animate()
    
    # Save results
    with open(os.path.join(cfg.logdir, 'results_cross_scene.json'), 'w') as f:
        json.dump(trainer.eval_metrics, f, indent=4)

if __name__ == '__main__':
    main()