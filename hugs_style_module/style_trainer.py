import torch
import torch.optim as optim
import torch
import torch.optim as optim
import torch.nn.functional as F

'''
python scripts/style_human.py   --cfg_file cfg_files/release/neuman/hugs_scene.yaml   --dataset_seq bike   --human_ckpt /path/to/human_final.pth   --style_text "Scream by Edvard Munch"   --steps 5000   --out_ckpt scene_stylized.pth
'''

from .clip_losses import (
    VGGContent,
    get_clip,
    content_loss_vgg,
    directional_clip_loss,
    patch_clip_loss,
    background_uniform_loss,
)

def run_style(
    adapter,
    style_text,
    steps=2000,
    lr=1e-3,
    lambda_d=5.0,
    lambda_p=90.0,
    lambda_c=0.8,
    lambda_b=10.0,
    out_ckpt="human_stylized.pth",
    device="cuda"
):
    device = torch.device(device)
    adapter.to(device)

    vgg = VGGContent(device)
    clip, tok = get_clip(device)
    print("Lamda values:", lambda_d, lambda_p, lambda_c, lambda_b)

    opt = optim.Adam(adapter.human_parameters(), lr=lr)

    for step in range(steps):
        idx = torch.randint(0, adapter.num_frames, (1,)).item()

        cam, pose = adapter.get_frame(idx)

        sty = adapter.render_human(cam, pose)

        # Handle 4-channel renders (drop alpha)
        if sty.shape[0] == 4:
            sty = sty[:3]

        # Resize + batch for CLIP/VGG
        sty_c = F.interpolate(sty.unsqueeze(0), size=(224, 224), mode='bilinear')
        src_c = torch.zeros_like(sty_c)

        # Losses
        L_c = content_loss_vgg(vgg, sty_c, src_c)
        L_d = directional_clip_loss(clip, tok, sty_c, src_c, text=style_text)
        L_p = patch_clip_loss(clip, tok, sty_c, src_c, text=style_text)
        L_b = background_uniform_loss(sty_c)

        loss = lambda_d * L_d + lambda_p * L_p + lambda_c * L_c + lambda_b * L_b

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            print(f"[{step+1}] loss={loss.item():.4f}")

    full_ckpt = adapter.model.state_dict()

    torch.save(full_ckpt, out_ckpt)
    print("Saved:", out_ckpt)


def self_state(adapter):
    # save only the appearance-decoder weights for safety
    return {
        "appearance_dec": adapter.model.appearance_dec.state_dict(),
    }
