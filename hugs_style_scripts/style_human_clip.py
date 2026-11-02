import argparse
import sys
from pathlib import Path
import torch

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))
from runtime import configure_runtime

configure_runtime()

from hugs_style_module.human_style_trainer import HumanGaussianAdapter, run_style_optimization
'''
python scripts/style_human.py   --cfg_file cfg_files/release/neuman/hugs_human.yaml   --dataset_seq bike   --human_ckpt /path/to/human_final.pth   --style_text "red clown"   --steps 30000   --out_ckpt human_stylized.pth
'''

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--style_text", type=str, default=None)
    p.add_argument("--style_image_path", type=str, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_d", type=float, default=5.0)
    p.add_argument("--lambda_p", type=float, default=90.0)
    p.add_argument("--lambda_c", type=float, default=0.8)
    p.add_argument("--lambda_b", type=float, default=10.0)
    p.add_argument("--lambda_rec", type=float, default=0.0)
    p.add_argument("--output_ckpt", type=str, default="hugs_stylized_human.pth")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_style_image(path, device):
    from PIL import Image
    from torchvision import transforms
    if path is None:
        return None
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
    ])
    return tfm(img).unsqueeze(0).to(device)


def main():
    args = parse_args()
    device = args.device

    adapter = HumanGaussianAdapter(
        cfg_file=args.cfg_file,
        checkpoint=args.checkpoint,
        device=device,
    )

    style_img = load_style_image(args.style_image_path, device) if args.style_image_path else None

    run_style_optimization(
        adapter=adapter,
        style_text=args.style_text,
        style_image=style_img,
        steps=args.steps,
        lr=args.lr,
        lambda_d=args.lambda_d,
        lambda_p=args.lambda_p,
        lambda_c=args.lambda_c,
        lambda_b=args.lambda_b,
        lambda_rec=args.lambda_rec,
        output_ckpt=args.output_ckpt,
        device=device,
    )


if __name__ == "__main__":
    main()
