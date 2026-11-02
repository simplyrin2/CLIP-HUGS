import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .clip_losses import (
    VGGContent,
    get_clip,
    content_loss_vgg,
    directional_clip_loss,
    patch_clip_loss,
    background_uniform_loss,
)


class HumanGaussianAdapter(nn.Module):
    """
    Thin wrapper around your HUGS model.

    You MUST implement:
      - __init__: load cfg + checkpoint, build model, dataset, etc.
      - num_frames: len(dataset) used for training
      - get_frame(idx): returns (gt_img, cam, pose)
      - render_human(cam, pose): returns human-only render [1,3,H,W] in [0,1]
      - human_parameters(): list of nn.Parameter of human Gaussians (pos, rot, scale, sh, opacity)
    """
    def __init__(self, cfg_file, checkpoint, device):
        super().__init__()
        self.device = device

        # TODO: load cfg, model, dataset, checkpoint using ml-hugs code.
        # Example structure (you will replace with real API):
        #
        #   from hugs.runner import build_model_and_dataset
        #   self.cfg, self.model, self.dataset = build_model_and_dataset(cfg_file)
        #   ckpt = torch.load(checkpoint, map_location=device)
        #   self.model.load_state_dict(ckpt["model"])
        #
        #   self.model.to(device).eval()  # for rendering
        #
        # You also need to know how to render "human-only" (HUGS already does this
        # for the human loss in the paper). Put that code into render_human.

        raise NotImplementedError("Implement HumanGaussianAdapter.__init__ for your repo.")

    @property
    def num_frames(self):
        # TODO: return number of training frames
        # e.g. return len(self.dataset["train"])
        raise NotImplementedError

    def get_frame(self, idx):
        """
        Returns:
          gt_img: [1,3,H,W] float32 in [0,1]
          cam: camera object or projection params
          pose: joint configuration for this frame
        """
        raise NotImplementedError

    def render_human(self, cam, pose):
        """
        Use HUGS renderer to render human-only image on solid background.

        Should return:
          img: [1,3,H,W] float32 in [0,1]
        """
        raise NotImplementedError

    def human_parameters(self):
        """
        Return list of nn.Parameter corresponding to human Gaussians only:
          - centers, rotations, scales, SH coeffs, opacity
        DO NOT include scene Gaussians or network weights you want frozen.
        """
        raise NotImplementedError


def run_style_optimization(
    adapter: HumanGaussianAdapter,
    style_text: str = None,
    style_image: torch.Tensor = None,
    steps: int = 2000,
    lr: float = 1e-3,
    lambda_d: float = 5.0,
    lambda_p: float = 90.0,
    lambda_c: float = 0.8,
    lambda_b: float = 10.0,
    lambda_rec: float = 0.0,
    output_ckpt: str = "hugs_stylized_human.pth",
    device: str = "cuda",
):
    device = torch.device(device)
    adapter.to(device)

    vgg = VGGContent(device)
    clip_model, tokenizer, _ = get_clip(device)

    if style_image is not None:
        style_image = style_image.to(device)

    params = adapter.human_parameters()
    optimizer = optim.Adam(params, lr=lr)

    for step in range(steps):
        idx = torch.randint(0, adapter.num_frames, (1,)).item()
        gt_img, cam, pose = adapter.get_frame(idx)
        gt_img = gt_img.to(device)

        sty_img = adapter.render_human(cam, pose)

        L_c = content_loss_vgg(vgg, sty_img, gt_img)
        L_d = directional_clip_loss(
            clip_model, tokenizer, sty_img, gt_img,
            style_text=style_text if style_image is None else None,
            style_img=style_image,
        )
        L_p = patch_clip_loss(
            clip_model, tokenizer, sty_img, gt_img,
            style_text=style_text if style_image is None else None,
            style_img=style_image,
        )
        L_b = background_uniform_loss(sty_img, bg_color=(0.0, 0.0, 0.0))

        loss = lambda_d * L_d + lambda_p * L_p + lambda_c * L_c + lambda_b * L_b

        # if lambda_rec > 0.0:
        #     # Optional: keep close to original reconstruction if you want
        #     with torch.no_grad():
        #         # TODO: implement full-frame rendering if desired
        #         pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(
                f"[{step+1}/{steps}] "
                f"Ld={L_d.item():.3f} Lp={L_p.item():.3f} "
                f"Lc={L_c.item():.3f} Lb={L_b.item():.3f} "
                f"tot={loss.item():.3f}"
            )

    # TODO: save updated human Gaussians into a checkpoint.
    # Typically you will get the underlying HUGS model object and call
    # torch.save({"model": model.state_dict(), ...}, output_ckpt)
    print(f"Finished style optimization. Save your model with updated human Gaussians to {output_ckpt}")
