import torch
import torch.nn as nn

from hugs.datasets.neuman import NeumanDataset
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.renderer.gs_renderer import render_human_scene


class HumanAdapter(nn.Module):
    def __init__(self, cfg_file, dataset_seq, checkpoint, device):
        super().__init__()
        self.device = device

        # 1) Dataset: cameras + SMPL params
        self.dataset = NeumanDataset(
            seq=dataset_seq,
            split="train",
            render_mode="human",
        )

        # 2) Init betas from dataset (like training code)
        import torch as _torch
        init_betas = _torch.stack(
            [x["betas"] for x in self.dataset.cached_data], dim=0
        )
        betas0 = init_betas[0]

        # 3) Human model (architecture must match hugs_human.yaml)
        self.model = HUGS_TRIMLP(
            sh_degree=0,
            n_subdivision=2,
            only_rgb=False,
            use_surface=False,
            init_2d=False,
            rotate_sh=False,
            isotropic=False,
            init_scale_multiplier=0.5,
            n_features=32,
            use_deformer=True,
            disable_posedirs=True,
            triplane_res=256,
            betas=betas0,
        )

        # 4) Load checkpoint (saved with human_gs.state_dict())
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

        self.model.load_state_dict(ckpt, cfg=None)  # cfg not needed for style

        # 5) Freeze everything
        # core Gaussian parameters (points)
        if isinstance(self.model._xyz, nn.Parameter):
            self.model._xyz.requires_grad = False

        # triplane + geometry + deformation decoders
        for p in self.model.triplane.parameters():
            p.requires_grad = False
        for p in self.model.geometry_dec.parameters():
            p.requires_grad = False
        for p in self.model.deformation_dec.parameters():
            p.requires_grad = False

        # 6) Unfreeze ONLY appearance decoder
        for p in self.model.appearance_dec.parameters():
            p.requires_grad = True

    @property
    def num_frames(self):
        return len(self.dataset)

    def get_frame(self, idx):
        data = self.dataset[idx]

        pose = {
            "global_orient": data["global_orient"],
            "body_pose": data["body_pose"],
            "betas": data["betas"],
            "transl": data["transl"],
            # training used smpl_scale[None]
            "smpl_scale": data["smpl_scale"][None],
        }
        return data, pose

    def render_human(self, data, pose):
        human_out = self.model.forward(
            global_orient=pose["global_orient"],
            body_pose=pose["body_pose"],
            betas=pose["betas"],
            transl=pose["transl"],
            smpl_scale=pose["smpl_scale"],
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
        )

        pkg = render_human_scene(
            data=data,
            human_gs_out=human_out,
            scene_gs_out=None,
            bg_color=torch.zeros(3, device=self.device),
            render_mode="human",
        )
        return pkg["render"]

    def human_parameters(self):
        # only style-related params
        return list(self.model.appearance_dec.parameters())
