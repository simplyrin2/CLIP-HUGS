import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import clip


class VGGContent(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.blocks = nn.Sequential(*list(vgg.children())[:16])
        for p in self.blocks.parameters():
            p.requires_grad = False
        self.blocks.eval().to(device)
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x):
        x = self.norm(x)
        return self.blocks(x)


def get_clip(device):
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, None  # OpenAI CLIP doesn't need separate tokenizer


def normalize_clip_img(x):
    # OpenAI CLIP normalization values
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device)[None,:,None,None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device)[None,:,None,None]
    return (x - mean) / std


def clip_image_embeds(model, x):
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    x = normalize_clip_img(x)
    return model.encode_image(x)



def clip_text_embeds(model, tokenizer, text):
    # OpenAI CLIP handles tokenization internally
    tok = clip.tokenize([text]).to(next(model.parameters()).device)
    return model.encode_text(tok)


def cosine_dist(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1 - (a * b).sum(dim=-1)


def content_loss_vgg(vgg, stylized, target):
    return F.mse_loss(vgg(stylized), vgg(target))


def directional_clip_loss(model, tokenizer, stylized, src, text=None):
    with torch.no_grad():
        src_f = clip_image_embeds(model, src)
        style_f = clip_text_embeds(model, tokenizer, text)
    sty_f = clip_image_embeds(model, stylized)
    dir_style = F.normalize(style_f - src_f, dim=-1)
    dir_sty = F.normalize(sty_f - src_f, dim=-1)
    return cosine_dist(dir_style, dir_sty).mean()

def patch_clip_loss(model, tokenizer, stylized, src, text=None,
                    patches=8, min_size=0.25, max_size=0.75):
    B, C, H, W = stylized.shape
    losses = []
    for _ in range(patches):
        rh = torch.randint(int(H*min_size), int(H*max_size)+1, (1,)).item()
        rw = torch.randint(int(W*min_size), int(W*max_size)+1, (1,)).item()
        y0 = torch.randint(0, H-rh+1, (1,)).item()
        x0 = torch.randint(0, W-rw+1, (1,)).item()
        patch_s = stylized[:,:,y0:y0+rh,x0:x0+rw]
        patch_t = src[:,:,y0:y0+rh,x0:x0+rw]

        patch_s = F.interpolate(patch_s, size=(224, 224), mode='bicubic', align_corners=False)
        patch_t = F.interpolate(patch_t, size=(224, 224), mode='bicubic', align_corners=False)

        losses.append(directional_clip_loss(model, tokenizer, patch_s, patch_t, text))
    return torch.stack(losses).mean()

def background_uniform_loss(img):
    return (img ** 2).mean()
