import torch
import torch.nn as nn

from hugs.datasets.neuman import NeumanDataset
from hugs.models import SceneGS
from hugs.models.hugs_trimlp import HUGS_TRIMLP
from hugs.renderer.gs_renderer import render_human_scene


class SceneAdapter(nn.Module):
    """Adapter for scene stylization while keeping human model frozen"""
    
    def __init__(self, cfg_file, dataset_seq, human_checkpoint, scene_checkpoint, device, scene_only=False):
        super().__init__()
        self.device = device

        # 1) Dataset: cameras + SMPL params
        self.dataset = NeumanDataset(
            seq=dataset_seq,
            split="train",
            render_mode="human_scene",  # Need both human and scene
        )

        # 2) Load human model (will be frozen) - only if not scene_only mode
        self.scene_only = scene_only
        self.human_model = None
        
        if not scene_only and human_checkpoint:
            init_betas = torch.stack(
                [x["betas"] for x in self.dataset.cached_data], dim=0
            )
            betas0 = init_betas[0]

            self.human_model = HUGS_TRIMLP(
                sh_degree=3,
                n_subdivision=0,
                use_surface=False,
                init_2d=False,
                rotate_sh=True,
                isotropic=False,
                init_scale_multiplier=1.0,
                n_features=32,
                use_deformer=True,
                disable_posedirs=False,
                triplane_res=256,
                betas=betas0,
            )

            # Load human checkpoint
            human_ckpt = torch.load(human_checkpoint, map_location=device, weights_only=False)
            self.human_model.load_state_dict(human_ckpt, cfg=None)
            
            # Freeze entire human model
            # HUGS_TRIMLP doesn't inherit from nn.Module, so we need to manually freeze each component
            for component in [self.human_model.triplane, self.human_model.geometry_dec, 
                            self.human_model.appearance_dec, self.human_model.deformation_dec]:
                if component is not None:
                    for p in component.parameters():
                        p.requires_grad = False
            # Set models to eval mode
            for component in [self.human_model.triplane, self.human_model.geometry_dec, 
                            self.human_model.appearance_dec, self.human_model.deformation_dec]:
                if component is not None and hasattr(component, 'eval'):
                    component.eval()

        # 3) Load scene model (will be trained for style)
        self.scene_model = SceneGS(sh_degree=3)
        
        # Load scene checkpoint
        scene_ckpt = torch.load(scene_checkpoint, map_location=device, weights_only=False)
        
        print(f"Scene checkpoint keys: {list(scene_ckpt.keys())}")
        
        # Use SceneGS restore method - handle different checkpoint formats
        try:
            # SceneGS.restore expects keys without underscore prefix
            if hasattr(self.scene_model, 'restore'):
                self.scene_model.restore(scene_ckpt, None)
            else:
                # Manual assignment if restore method not available
                if '_xyz' in scene_ckpt:
                    # Direct state format with underscores
                    self.scene_model._xyz = scene_ckpt['_xyz']
                    self.scene_model._features_dc = scene_ckpt['_features_dc']
                    self.scene_model._features_rest = scene_ckpt['_features_rest']
                    self.scene_model._scaling = scene_ckpt['_scaling']
                    self.scene_model._rotation = scene_ckpt['_rotation']
                    self.scene_model._opacity = scene_ckpt['_opacity']
                    self.scene_model.active_sh_degree = scene_ckpt['active_sh_degree']
                    self.scene_model.max_radii2D = scene_ckpt['max_radii2D']
                    self.scene_model.xyz_gradient_accum = scene_ckpt['xyz_gradient_accum']
                    self.scene_model.denom = scene_ckpt['denom']
                elif 'xyz' in scene_ckpt:
                    # Format without underscores  
                    self.scene_model._xyz = scene_ckpt['xyz']
                    self.scene_model._features_dc = scene_ckpt['features_dc']
                    self.scene_model._features_rest = scene_ckpt['features_rest']
                    self.scene_model._scaling = scene_ckpt['scaling']
                    self.scene_model._rotation = scene_ckpt['rotation']
                    self.scene_model._opacity = scene_ckpt['opacity']
                    self.scene_model.active_sh_degree = scene_ckpt['active_sh_degree']
                    self.scene_model.max_radii2D = scene_ckpt['max_radii2D']
                    self.scene_model.xyz_gradient_accum = scene_ckpt['xyz_gradient_accum']
                    self.scene_model.denom = scene_ckpt['denom']
                else:
                    raise ValueError(f"Unknown checkpoint format. Keys: {list(scene_ckpt.keys())}")
        except Exception as e:
            print(f"Error loading scene checkpoint: {e}")
            print("Available checkpoint keys:", list(scene_ckpt.keys()))
            print("Trying manual assignment...")
            
            # Last resort: try both formats
            for prefix in ['', '_']:
                try:
                    xyz_key = f'{prefix}xyz'
                    if xyz_key in scene_ckpt:
                        self.scene_model._xyz = scene_ckpt[f'{prefix}xyz']
                        self.scene_model._features_dc = scene_ckpt[f'{prefix}features_dc']
                        self.scene_model._features_rest = scene_ckpt[f'{prefix}features_rest']
                        self.scene_model._scaling = scene_ckpt[f'{prefix}scaling']
                        self.scene_model._rotation = scene_ckpt[f'{prefix}rotation']
                        self.scene_model._opacity = scene_ckpt[f'{prefix}opacity']
                        self.scene_model.active_sh_degree = scene_ckpt['active_sh_degree']
                        self.scene_model.max_radii2D = scene_ckpt['max_radii2D']
                        self.scene_model.xyz_gradient_accum = scene_ckpt['xyz_gradient_accum']
                        self.scene_model.denom = scene_ckpt['denom']
                        print(f"Successfully loaded using '{prefix}' prefix format")
                        break
                except KeyError:
                    continue
            else:
                raise ValueError(f"Could not load scene checkpoint with any known format")

        # 4) Setup scene parameters for training
        # Only train appearance-related parameters for style transfer
        self.scene_model._xyz.requires_grad = False  # Keep geometry fixed
        self.scene_model._features_dc.requires_grad = True  # Allow color changes
        self.scene_model._features_rest.requires_grad = True  # Allow SH changes
        self.scene_model._scaling.requires_grad = False  # Keep scale fixed
        self.scene_model._rotation.requires_grad = False  # Keep rotation fixed
        self.scene_model._opacity.requires_grad = False  # Keep opacity fixed

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
            "smpl_scale": data["smpl_scale"][None],
        }
        return data, pose

    def render_scene_only(self, data):
        """Render only the scene (no human)"""
        scene_out = self.scene_model.forward()
        
        pkg = render_human_scene(
            data=data,
            human_gs_out=None,
            scene_gs_out=scene_out,
            bg_color=torch.zeros(3, device=self.device),
            render_mode='scene'
        )
        
        return pkg["render"]

    def render_human_scene(self, data, pose):
        """Render both human and scene together (or scene only if scene_only=True)"""
        if self.scene_only or self.human_model is None:
            # Scene-only rendering
            return self.render_scene_only(data)
        
        # Get human output (frozen)
        with torch.no_grad():
            human_out = self.human_model.forward(
                global_orient=pose["global_orient"],
                body_pose=pose["body_pose"],
                betas=pose["betas"],
                transl=pose["transl"],
                smpl_scale=pose["smpl_scale"],
                dataset_idx=-1,
                is_train=False,
                ext_tfs=None,
            )
        
        # Get scene output (trainable)
        scene_out = self.scene_model.forward()

        pkg = render_human_scene(
            data=data,
            human_gs_out=human_out,
            scene_gs_out=scene_out,
            bg_color=torch.zeros(3, device=self.device),
            render_mode='human_scene'
        )

        return pkg["render"]

    def scene_parameters(self):
        """Return only the scene parameters that should be optimized"""
        params = []
        if self.scene_model._features_dc.requires_grad:
            params.append(self.scene_model._features_dc)
        if self.scene_model._features_rest.requires_grad:
            params.append(self.scene_model._features_rest)
        return params

    def save_scene_checkpoint(self, filepath):
        """Save scene model state using SceneGS.state_dict() format"""
        # Use the same format as SceneGS.state_dict() - keys without underscores
        scene_state = {
            'active_sh_degree': self.scene_model.active_sh_degree,
            'xyz': self.scene_model._xyz,
            'features_dc': self.scene_model._features_dc,
            'features_rest': self.scene_model._features_rest,
            'scaling': self.scene_model._scaling,
            'rotation': self.scene_model._rotation,
            'opacity': self.scene_model._opacity,
            'max_radii2D': self.scene_model.max_radii2D,
            'xyz_gradient_accum': self.scene_model.xyz_gradient_accum,
            'denom': self.scene_model.denom,
            'optimizer': self.scene_model.optimizer.state_dict() if hasattr(self.scene_model, 'optimizer') and self.scene_model.optimizer is not None else {},
            'spatial_lr_scale': getattr(self.scene_model, 'spatial_lr_scale', 1.0),
        }
        torch.save(scene_state, filepath)
        print(f"Saved scene checkpoint with SceneGS-compatible keys: {list(scene_state.keys())}")
        return filepath