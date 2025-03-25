""" 
This module implements a renderer that combines appearance embedding with per-pixel visibility map estimation.
It is built on top of the GSplatV1Renderer, reusing its efficient projection and rasterization pipeline while
adding a separate network to predict a transient (visibility) signal.
"""

from typing import Tuple, Optional, Any, List
from dataclasses import dataclass, field
import lightning
import torch
from torch import nn
import tinycudann as tcnn
from internal.encodings.positional_encoding import PositionalEncoding
from internal.utils.network_factory import NetworkFactory
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from .gsplat_v1_renderer import GSplatV1Renderer, GSplatV1, GSplatV1RendererModule, spherical_harmonics, spherical_harmonics_decomposed


from . import RendererOutputInfo, RendererOutputTypes
from .renderer import Renderer, RendererConfig
# --- Configuration dataclasses ---

@dataclass
class UVEncodingConfig:
    n_levels: int = 8
    base_resolution: int = 16
    per_level_scale: float = 1.405

@dataclass
class NetworkConfig:
    n_neurons: int = 64
    n_layers: int = 3
    skip_layers: List[int] = field(default_factory=lambda: [])

@dataclass
class AppearanceNetworkConfig(NetworkConfig):
    pass

@dataclass
class VisibilityNetworkConfig(NetworkConfig):
    pass

@dataclass
class ModelConfig:
    n_images: int = -1
    # appearance branch
    n_gaussian_feature_dims: int = 64
    n_appearance_embedding_dims: int = 128
    appearance_network: AppearanceNetworkConfig = field(default_factory=AppearanceNetworkConfig)
    is_view_dependent: bool = False
    n_view_direction_frequencies: int = 4
    # transient branch for visibility estimation
    n_transient_embedding_dims: int = 128
    uv_encoding: UVEncodingConfig = field(default_factory=UVEncodingConfig)
    visibility_network: VisibilityNetworkConfig = field(default_factory=VisibilityNetworkConfig)
    with_opacity: bool = False

    
    tcnn: bool = False 
@dataclass
class LRConfig:
    lr_init: float
    lr_final_factor: float = 0.1

@dataclass
class OptimizationConfig:
    gamma_eps: float = 1e-6
    appearance_embedding: LRConfig = field(default_factory=lambda: LRConfig(lr_init=2e-3))
    appearance_network: LRConfig = field(default_factory=lambda: LRConfig(lr_init=1e-3))
    transient_embedding: LRConfig = field(default_factory=lambda: LRConfig(lr_init=2e-3))
    uv_encoding: LRConfig = field(default_factory=lambda: LRConfig(lr_init=2e-3))
    visibility_network: LRConfig = field(default_factory=lambda: LRConfig(lr_init=1e-3))
    eps: float = 1e-15
    max_steps: int = 30000
    appearance_warm_up: int = 1000
    transient_warm_up: int = 2000

# --- Model that holds the appearance and visibility networks ---

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._setup()

    def _setup(self):
        network_factory = NetworkFactory(tcnn=self.config.tcnn)
        # Appearance branch
        self.appearance_embedding = nn.Embedding(
            num_embeddings=self.config.n_images,
            embedding_dim=self.config.n_appearance_embedding_dims,
        )
        n_app_in = self.config.n_gaussian_feature_dims + self.config.n_appearance_embedding_dims
        if self.config.is_view_dependent:
            self.view_direction_encoding = PositionalEncoding(3, self.config.n_view_direction_frequencies)
            n_app_in += self.view_direction_encoding.get_output_n_channels()
        self.appearance_network = network_factory.get_network_with_skip_layers(
            n_input_dims=n_app_in,
            n_output_dims=3,
            n_layers=self.config.appearance_network.n_layers,
            n_neurons=self.config.appearance_network.n_neurons,
            activation="ReLU",
            output_activation="Sigmoid",
            skips=self.config.appearance_network.skip_layers,
        )
        # Transient branch (for visibility map)
        self.transient_embedding = nn.Embedding(
            num_embeddings=self.config.n_images,
            embedding_dim=self.config.n_transient_embedding_dims,
        )
        # Create a UV encoding for each image
        self.uv_encodings = nn.ModuleList()
        for i in range(self.config.n_images):
            self.uv_encodings.append(tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "DenseGrid",
                    "n_levels": self.config.uv_encoding.n_levels,
                    "base_resolution": self.config.uv_encoding.base_resolution,
                    "per_level_scale": self.config.uv_encoding.per_level_scale,
                },
                seed=i,
                dtype=torch.float,
            ))
        n_vis_in = self.config.n_transient_embedding_dims + self.uv_encodings[0].n_output_dims
        self.visibility_network = network_factory.get_network_with_skip_layers(
            n_input_dims=n_vis_in,
            n_output_dims=1,
            n_layers=self.config.visibility_network.n_layers,
            n_neurons=self.config.visibility_network.n_neurons,
            activation="ReLU",
            output_activation="Sigmoid",
        )

    def appearance_forward(self, gaussian_features, appearance, view_dirs):
        app_embed = self.appearance_embedding(appearance.reshape((-1,))).repeat(gaussian_features.shape[0], 1)
        input_list = [gaussian_features, app_embed]
        if self.config.is_view_dependent:
            input_list.append(self.view_direction_encoding(view_dirs))
        app_input = torch.concat(input_list, dim=-1)
        return self.appearance_network(app_input)

    def visibility_forward(self, width: int, height: int, appearance):
        n = width * height
        transient_embed = self.transient_embedding(appearance.reshape((-1,))).repeat(n, 1)
        grid_x, grid_y = torch.meshgrid(
            torch.arange(width, dtype=torch.float, device=transient_embed.device),
            torch.arange(height, dtype=torch.float, device=transient_embed.device),
            indexing="xy",
        )
        grid_norm = torch.concat([grid_x.unsqueeze(-1) / (width - 1), grid_y.unsqueeze(-1) / (height - 1)], dim=-1)
        uv_input = grid_norm.reshape((-1, 2))
        uv_out = self.uv_encodings[appearance.item()](uv_input)
        vis_input = torch.concat([uv_out, transient_embed], dim=-1)
        return self.visibility_network(vis_input).reshape(grid_norm.shape[:-1])

    def forward(self, width: int, height: int, gaussian_features, appearance, view_dirs):
        # print(f'Shape of gaussian filters in forward of model:{gaussian_features.shape}')
        app_out = self.appearance_forward(gaussian_features, appearance, view_dirs)
        vis_out = self.visibility_forward(width, height, appearance)
        return app_out, vis_out
    
@dataclass
class GSplatAppearanceEmbeddingVisibilityMapRenderer(GSplatV1Renderer):
    separate_sh: bool = True

    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "GSplatAppearanceEmbeddingVisibilityMapRendererModule":
        assert self.separate_sh is True
        if getattr(self, "model_config", None) is not None:
            # checkpoint generated by previous version
            self.model = self.config.model
            self.optimization = self.config.optimization        
        
        return GSplatAppearanceEmbeddingVisibilityMapRendererModule(self)


class GSplatAppearanceEmbeddingVisibilityMapRendererModule(GSplatV1RendererModule):
    """
    This renderer module first uses the GSplatV1 pipeline to project and rasterize
    the gaussian model. Then, it uses the model module (holding both the appearance
    and transient networks) to compute an RGB offset and a visibility map.
    """

    def setup(self, stage: str, lightning_module=None, *args: Any, **kwargs: Any) -> Any:
        if lightning_module is not None:
            if self.config.model.n_images <= 0:
                max_input_id = 0
                appearance_group_ids = lightning_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
                if appearance_group_ids is not None:
                    for i in appearance_group_ids.values():
                        if i[0] > max_input_id:
                            max_input_id = i[0]
                self.config.model.n_images = max_input_id + 1
            self._setup_model()
            # print(self.model)

    def _setup_model(self, device=None):
        # Import Model from its module – adjust the import as needed.
        self.model = Model(self.config.model)
        if device is not None:
            self.model.to(device=device)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.config.model.n_images = state_dict["model.appearance_embedding.weight"].shape[0]
        self._setup_model(device=state_dict["model.appearance_embedding.weight"].device)
        return super().load_state_dict(state_dict, strict)

    def training_setup(self, module: lightning.LightningModule):
        app_emb_opt, app_emb_sched = self._create_optimizer_and_scheduler(
            self.model.appearance_embedding.parameters(),
            "appearance_embedding",
            lr_init=self.config.optimization.appearance_embedding.lr_init,
            lr_final_factor=self.config.optimization.appearance_embedding.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.appearance_warm_up,
        )
        app_net_opt, app_net_sched = self._create_optimizer_and_scheduler(
            self.model.appearance_network.parameters(),
            "appearance_network",
            lr_init=self.config.optimization.appearance_network.lr_init,
            lr_final_factor=self.config.optimization.appearance_network.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.appearance_warm_up,
        )
        trans_emb_opt, trans_emb_sched = self._create_optimizer_and_scheduler(
            self.model.transient_embedding.parameters(),
            "transient_embedding",
            lr_init=self.config.optimization.transient_embedding.lr_init,
            lr_final_factor=self.config.optimization.transient_embedding.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.transient_warm_up,
        )
        uv_opt, uv_sched = self._create_optimizer_and_scheduler(
            self.model.uv_encodings.parameters(),
            "uv_encoding",
            lr_init=self.config.optimization.uv_encoding.lr_init,
            lr_final_factor=self.config.optimization.uv_encoding.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.transient_warm_up,
        )
        vis_net_opt, vis_net_sched = self._create_optimizer_and_scheduler(
            self.model.visibility_network.parameters(),
            "visibility_network",
            lr_init=self.config.optimization.visibility_network.lr_init,
            lr_final_factor=self.config.optimization.visibility_network.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.transient_warm_up,
        )

        if self.config.model.with_opacity:
            module.extra_train_metrics.append(self.opacity_offset_reg)

        return (
            [app_emb_opt, app_net_opt, trans_emb_opt, uv_opt, vis_net_opt],
            [app_emb_sched, app_net_sched, trans_emb_sched, uv_sched, vis_net_sched]
        )
    
    def sh(self, pc, dirs, mask=None):
        if pc.is_pre_activated:
            return spherical_harmonics(
                pc.active_sh_degree,
                dirs,
                pc.get_shs(),
                masks=mask,
            )
        return spherical_harmonics_decomposed(
            pc.active_sh_degree,
            dirs,
            dc=pc.get_shs_dc(),
            coeffs=pc.get_shs_rest(),
            masks=mask,
        )
    
    def selective_sh(self, pc, dirs, mask):
        if pc.is_pre_activated:
            return spherical_harmonics(
                pc.active_sh_degree,
                dirs,
                pc.get_shs()[mask],
            )
        return spherical_harmonics_decomposed(
            pc.active_sh_degree,
            dirs,
            dc=pc.get_shs_dc()[mask],
            coeffs=pc.get_shs_rest()[mask],
        )    
    
    def get_rgbs(
        self,
        camera,
        pc,
        projections: Tuple,
        visibility_filter,
        status: Any,
        **kwargs,
    ):
        """
        This is where we handle skip_appearance. The parent class calls self.get_rgb
        for each visible Gaussian.
        """
        skip_appearance = kwargs.get("skip_appearance", False)
        skip_visibility = kwargs.get("skip_visibility", False)
        print(f'Inside forward get rgb: Skip appearance:{skip_appearance}, skip visibility: {skip_visibility}')
        # print(f'visiblity filter shape:{visibility_filter.shape}')
        # true_count = torch.sum(visibility_filter).item()
        # print(f'True count for visibility:{true_count}')
        # 1) base color from spherical harmonics

        # calculate normalized view directions
        detached_xyz = pc.get_xyz.detach()[visibility_filter]
        view_directions = detached_xyz - camera.camera_center  # (N, 3)
        view_directions = torch.nn.functional.normalize(view_directions, dim=-1)

        base_rgb = self.selective_sh(
                pc,
                view_directions,
                visibility_filter,
                ) + 0.5

        if skip_appearance and skip_visibility:
            # Warm-up: no offset from MLP
            print(f'I am in first if')
            # print(f'base_rgb shape:{base_rgb.shape},other tensor shape:{torch.ones(1,camera.height,camera.width)}')
            rgb = torch.clamp(
                self.sh(
                    pc,
                    pc.get_xyz.detach() - camera.camera_center,
                    visibility_filter,
                ) + 0.5,
                min=0.,
            )
            return rgb, torch.ones(1, camera.height, camera.width)
        elif skip_appearance==True and skip_visibility==False:
            print(f'I am in second if')
            _, visibility_from_mlp = self.model(
                camera.width.item(),
                camera.height.item(),
                pc.get_appearance_features()[visibility_filter],
                camera.appearance_id,
                view_directions
            )
            rgb = torch.clamp(
                self.sh(
                    pc,
                    pc.get_xyz.detach() - camera.camera_center,
                    visibility_filter,
                ) + 0.5,
                min=0.,
            )
            return torch.clamp(rgb, 0., 1.), visibility_from_mlp.unsqueeze(0)       
        elif skip_appearance==False and skip_visibility==True:
            print(f'I am in third if')
            raw_rgb_offset, _ = self.model(
                camera.width.item(),
                camera.height.item(),
                pc.get_appearance_features()[visibility_filter],
                camera.appearance_id,
                view_directions
            )

            rgb_offset = raw_rgb_offset * 2.0 - 1.0
   
            means2d = projections[1]
            rgbs = torch.zeros((pc.n_gaussians, 3), dtype=means2d.dtype, device=means2d.device)
            rgbs[visibility_filter] = torch.clamp(
                base_rgb + rgb_offset,
                min=0.,
                max=1.,
            )
            return rgbs, torch.ones(1,camera.height,camera.width)
        
        else:
            raw_rgb_offset, visibility_from_mlp = self.model(
                camera.width.item(),
                camera.height.item(),
                pc.get_appearance_features()[visibility_filter],
                camera.appearance_id,
                view_directions
            )

            rgb_offset = raw_rgb_offset * 2.0 - 1.0
            means2d = projections[1]
            rgbs = torch.zeros((pc.n_gaussians, 3), dtype=means2d.dtype, device=means2d.device)
            rgbs[visibility_filter] = torch.clamp(
                base_rgb + rgb_offset,
                min=0.,
                max=1.,
            )
            return rgbs, visibility_from_mlp.unsqueeze(0)

    # def get_rgb(self, pc, gaussian_indices, viewdirs, camera: Camera, **kwargs):
    #     """
    #     This is where we handle skip_appearance. The parent class calls self.get_rgb
    #     for each visible Gaussian.
    #     """
    #     skip_appearance = kwargs.get("skip_appearance", False)

    #     # 1) base color from spherical harmonics
    #     base_rgb = spherical_harmonics(
    #         pc.active_sh_degree,
    #         viewdirs,
    #         pc.get_features[gaussian_indices]
    #     ) + 0.5

    #     if skip_appearance:
    #         # Warm-up: no offset from MLP
    #         return torch.clamp(base_rgb, 0., 1.)

    #     # 2) otherwise do normal appearance offset
    #     offset = self.appearance_model(
    #         pc.get_appearance_features()[gaussian_indices],
    #         camera.appearance_id,
    #         viewdirs
    #     )  # presumably in [0,1]
    #     offset = offset * 2.0 - 1.0
    #     return torch.clamp(base_rgb + offset, 0., 1.)


    def forward(self, viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        render_type_bits = self.parse_render_types(render_types)

        preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)

        # 1. get scales and then project
        scales, status = self.get_scales(viewpoint_camera, pc, **kwargs)
        if scaling_modifier != 1.:
            scales = scales * scaling_modifier

        projections = GSplatV1.project(
            preprocessed_camera,
            pc.get_means(),
            scales,
            pc.get_rotations(),
            eps2d=self.config.filter_2d_kernel_size,
            anti_aliased=self.config.anti_aliased,
        )
        radii, means2d, depths, conics, compensations = projections

        radii_squeezed = radii.squeeze(0)
        visibility_filter = radii_squeezed > 0

        # 2. get opacities and then isect encoding
        opacities, status = self.get_opacities(
            viewpoint_camera,
            pc,
            projections,
            visibility_filter,
            status,
            **kwargs,
        )

        opacities = opacities.unsqueeze(0)  # [1, N]
        if self.config.anti_aliased:
            opacities = opacities * compensations

        isects = self.isect_encode(
            preprocessed_camera,
            projections,
            opacities,
            tile_size=self.config.block_size,
        )

        # 3. rasterization
        means2d = means2d.squeeze(0)
        projection_for_rasterization = radii, means2d, depths, conics, compensations

        def rasterize(input_features: torch.Tensor, background, return_alpha: bool = False):
            rendered_colors, rendered_alphas = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities,
                colors=input_features,
                background=background,
                tile_size=self.config.block_size,
            )

            if return_alpha:
                return rendered_colors, rendered_alphas.squeeze(0).squeeze(-1)
            return rendered_colors

        # rgb
        rgb = None
        visibility_from_mlp = None
        if self.is_type_required(render_type_bits, self._RGB_REQUIRED):
            rgbs, visibility_from_mlp = self.get_rgbs(
                viewpoint_camera,
                pc,
                projections,
                visibility_filter,
                status,
                **kwargs,
            )
            rgb = rasterize(rgbs, bg_color).permute(2, 0, 1)
            visibility_from_mlp = visibility_from_mlp.to(device=bg_color.device)
        alpha = None
        acc_depth_im = None
        acc_depth_inverted_im = None
        exp_depth_im = None
        exp_depth_inverted_im = None
        inv_depth_alt = None
        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            # acc depth
            acc_depth_im, alpha = rasterize(depths[0].unsqueeze(-1), torch.zeros((1,), device=bg_color.device), True)
            alpha = alpha[..., None]

            # acc depth inverted
            if self.is_type_required(render_type_bits, self._ACC_DEPTH_INVERTED_REQUIRED):
                acc_depth_inverted_im = torch.where(acc_depth_im > 0, 1. / acc_depth_im, acc_depth_im.detach().max())
                acc_depth_inverted_im = acc_depth_inverted_im.permute(2, 0, 1)

            # exp depth
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_REQUIRED):
                exp_depth_im = torch.where(alpha > 0, acc_depth_im / alpha, acc_depth_im.detach().max())

                exp_depth_im = exp_depth_im.permute(2, 0, 1)

            # alpha
            if self.is_type_required(render_type_bits, self._ALPHA_REQUIRED):
                alpha = alpha.permute(2, 0, 1)
            else:
                alpha = None

            # permute acc depth
            acc_depth_im = acc_depth_im.permute(2, 0, 1)

            # exp depth inverted
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_INVERTED_REQUIRED):
                exp_depth_inverted_im = torch.where(exp_depth_im > 0, 1. / exp_depth_im, exp_depth_im.detach().max())

        # inverse depth
        inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths[0].clamp_min(0.) + 1e-8).unsqueeze(-1)
            inverse_depth_im = rasterize(inverse_depth, torch.zeros((1,), dtype=torch.float, device=bg_color.device)).permute(2, 0, 1)
            inv_depth_alt = inverse_depth_im

        # hard depth
        hard_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_DEPTH_REQUIRED):
            hard_depth_im, _ = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities + (1 - opacities.detach()),
                colors=depths[0].unsqueeze(-1),
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                tile_size=self.config.block_size,
            )
            hard_depth_im = hard_depth_im.permute(2, 0, 1)

        # hard inverse depth
        hard_inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths[0].clamp_min(0.) + 1e-8).unsqueeze(-1)
            hard_inverse_depth_im, _ = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities + (1 - opacities.detach()),
                colors=inverse_depth,
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                tile_size=self.config.block_size,
            )

            hard_inverse_depth_im = hard_inverse_depth_im.permute(2, 0, 1)
            inv_depth_alt = hard_inverse_depth_im

        return {
            "render": rgb,
            "alpha": alpha,
            "acc_depth": acc_depth_im,
            "acc_depth_inverted": acc_depth_inverted_im,
            "exp_depth": exp_depth_im,
            "exp_depth_inverted": exp_depth_inverted_im,
            "inverse_depth": inverse_depth_im,
            "hard_depth": hard_depth_im,
            "hard_inverse_depth": hard_inverse_depth_im,
            "inv_depth_alt": inv_depth_alt,
            "viewspace_points": means2d,
            "viewspace_points_grad_scale": 0.5 * torch.tensor([preprocessed_camera[-1]]).to(means2d),
            "visibility_filter": visibility_filter,
            "radii": radii_squeezed,
            "scales": scales,
            "opacities": opacities[0],
            "projections": projections,
            "isects": isects,
            "visibility": visibility_from_mlp,
            "extra_image": visibility_from_mlp.squeeze(0)
        }

    # def forward(self, viewpoint_camera, pc, bg_color, scaling_modifier=1.0, skip_appearance=False, skip_transient=False, **kwargs):
    #     """
    #     Overridden forward to attach visibility map or do normal pass.
    #     We'll still rely on the parent's tile-based culling, but we
    #     override get_rgb to check skip_appearance, and also manually handle
    #     skip_transient for the final output.
    #     """
    #     # Use parent's forward, but it will call self.get_rgb
    #     # so our get_rgb can see skip_appearance
    #     outputs = super().forward(viewpoint_camera, pc, bg_color, scaling_modifier=scaling_modifier, **kwargs)

    #     # If we haven't warmed up transient logic, override visibility
    #     if skip_transient or skip_appearance:
    #         # i.e. fill with 1.0 so everything is fully visible
    #         h, w = viewpoint_camera.height.item(), viewpoint_camera.width.item()
    #         outputs["visibility"] = torch.ones((1, h, w), device=bg_color.device)
    #         outputs["extra_image"] = outputs["visibility"]
    #     else:
    #         # if not skipping transient, we might call a function that
    #         # produces the real per-pixel visibility map
    #         # e.g.:
    #         raw_rgb_offset, _ = self.model(
    #             viewpoint_camera.width.item(),
    #             viewpoint_camera.height.item(),
    #             pc.get_appearance_features()[gaussian_indices],
    #             viewpoint_camera.appearance_id,
    #             viewdirs
    #         )
    #         _,vis_map = self.compute_transient_visibility(viewpoint_camera, pc, **kwargs)
    #         outputs["visibility"] = vis_map
    #         outputs["extra_image"] = vis_map




    #     return outputs


    def training_forward(self, step: int, module, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, **kwargs):
        """
        The main place to toggle the warm-ups for appearance vs. transient.
        """
        # 1) read thresholds from config
        appearance_warm_up = self.config.optimization.appearance_warm_up
        transient_warm_up = self.config.optimization.transient_warm_up

        # 2) if step < appearance_warm_up, skip advanced appearance offset
        skip_appearance = (step < appearance_warm_up)

        # 3) if step < transient_warm_up, skip transient logic (use all-ones visibility)
        skip_transient = (step < transient_warm_up)
        print(f'I am in training forward at step: {step}: Skip appearance:{skip_appearance}, skip visibility: {skip_transient}')

        # 4) call forward with flags
        render_outputs = self.forward(
            viewpoint_camera,
            pc,
            bg_color,
            scaling_modifier,
            skip_appearance=skip_appearance,
            skip_transient=skip_transient,
            **kwargs
        )

        return render_outputs


    def compute_transient_visibility(self, viewpoint_camera, pc, **kwargs):
        """
        Suppose you have an MLP or stored 2D map. For example:
        """
        h, w = viewpoint_camera.height.item(), viewpoint_camera.width.item()
        # placeholder example: random mask
        # you'd likely do something more sophisticated here
        return torch.rand((1, h, w), device=pc.get_xyz.device)

    @staticmethod
    def _create_optimizer_and_scheduler(params, name, lr_init, lr_final_factor, max_steps, eps, warm_up) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            params=[{"params": list(params), "name": name}],
            lr=lr_init,
            eps=eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: lr_final_factor ** min(max(iter - warm_up, 0) / max_steps, 1),
            verbose=False,
        )
        return optimizer, scheduler
    



