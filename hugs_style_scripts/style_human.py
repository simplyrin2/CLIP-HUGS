import argparse
import sys
import os
# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import configure_runtime

configure_runtime()

from hugs_style_module.human_adapter import HumanAdapter
from hugs_style_module.style_trainer import run_style


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", required=True)
    p.add_argument("--dataset_seq", required=True)
    p.add_argument("--human_ckpt", required=True)
    p.add_argument("--style_text", required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_ckpt", default="human_stylized.pth")
    args = p.parse_args()

    adapter = HumanAdapter(
        cfg_file=args.cfg_file,
        dataset_seq=args.dataset_seq,
        checkpoint=args.human_ckpt,
        device=args.device
    )

    run_style(
        adapter=adapter,
        style_text=args.style_text,
        steps=args.steps,
        lr=args.lr,
        out_ckpt=args.out_ckpt,
        device=args.device
    )


if __name__ == "__main__":
    main()
