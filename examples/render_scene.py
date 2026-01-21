import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import assert_never

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
import viser
from datasets.colmap import Dataset, Parser, BlenderDataset
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
)

from textured_gaussians.rendering import rasterization_2dgs, rasterization_textured_gaussians
from textured_gaussians.strategy import DefaultStrategy, MCMCStrategy

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = "../outputs/old_hall/depth_scale/ckpts/ckpt_29999.pt"
    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "../data/old_hall/train"
    # Directory to save results
    result_dir: str = "output/1"

    # Dataset mode
    dataset: str = "colmap"

    # Downsample factor for the dataset
    data_factor: int = 1
    
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    min_opacity: float = 0.005

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    background_mode: str = None

    # Enable camera optimization.
    pose_opt: bool = False
    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Enable depth loss. (experimental)
    depth_loss: bool = False

    # scale_loss
    scale_loss: bool = False 
    scale_lambda: float = 1e-1
    
    # Model for splatting.
    model_type: Literal["2dgs", "textured_gaussians"] = "2dgs"

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=MCMCStrategy
    )

    # Pretrained checkpoints
    pretrained_path: str = None

    # textured gaussians
    texture_resolution: int = 50

class Runner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = "cuda"
        self.model_type = cfg.model_type
        # Load data: Training data should contain initial points and colors.
        if cfg.dataset == "colmap":
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=True,
                test_every=cfg.test_every,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
            self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        elif cfg.dataset == "blender":
            self.parser = None
            if cfg.background_mode == "white":
                bg_color = (255, 255, 255)
            else:
                bg_color = (0, 0, 0)
            self.trainset = BlenderDataset(data_dir=cfg.data_dir, split="train", bg_color=bg_color)
            self.valset = BlenderDataset(data_dir=cfg.data_dir, split="val", bg_color=bg_color)
            self.scene_scale = 1.0 # no scaling required
        else:
            raise ValueError(f"Dataset mode {cfg.dataset} not supported!")


        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            self.cfg,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )

        self.server = viser.ViserServer(port=cfg.port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        if cfg.dataset == "colmap":
            camtoworlds = self.parser.camtoworlds[5:-5]
            camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
            camtoworlds = np.concatenate(
                [
                    camtoworlds,
                    np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
                ],
                axis=1,
            )  # [N, 4, 4]

            camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
            K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
            width, height = list(self.parser.imsize_dict.values())[0]
        elif cfg.dataset == "blender":
            camtoworlds = np.stack(self.trainset.camtoworlds) # [N, 4, 4]
            camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
            camtoworlds = np.concatenate(
                [
                    camtoworlds,
                    np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
                ],
                axis=1,
            )  # [N, 4, 4]
            camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
            K = torch.from_numpy(self.trainset.K).float().to(device)
            width, height = self.trainset.image_size, self.trainset.image_size

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _, surf_normals, _, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=1 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _, _, _, _, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]

        opacities = torch.sigmoid(self.splats["opacities"]) # [N,]
        

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        if self.model_type == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
                info,
            ) = rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "textured_gaussians":
            textures = self.get_textures()
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
                info,
            ) = rasterization_textured_gaussians(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                textures=textures,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            _,
            _,
            info,
        )

def create_splats_with_optimizers(
    parser: Parser,
    cfg: Config,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        if init_num_pts < points.shape[0]:
            sampled_pts_idx = np.random.choice(points.shape[0], init_num_pts, replace=False)
        else:
            sampled_pts_idx = np.arange(points.shape[0])
        # randomly sample points from the SfM points
        points = points[sampled_pts_idx]
        rgbs = rgbs[sampled_pts_idx]
    elif init_type == "pretrained":
        assert cfg.pretrained_path is not None
        ckpt = torch.load(cfg.pretrained_path)["splats"]
        if init_num_pts < ckpt["means"].shape[0]:
            sampled_pts_idx = np.random.choice(ckpt["means"].shape[0], init_num_pts, replace=False)
        else:
            sampled_pts_idx = np.arange(ckpt["means"].shape[0])
        points = ckpt["means"][sampled_pts_idx]
        rgbs = torch.rand((points.shape[0], 3))
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    
    if init_type == "pretrained":
        scales = ckpt["scales"][sampled_pts_idx]
        quats = ckpt["quats"][sampled_pts_idx]
        opacities = ckpt["opacities"][sampled_pts_idx]
    else:
        N = points.shape[0]
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    # SH coefficients
    if feature_dim is None:
        # color is SH coefficients.
        if init_type == "pretrained":
            params.append(("sh0", torch.nn.Parameter(ckpt["sh0"][sampled_pts_idx]), 2.5e-3))
            params.append(("shN", torch.nn.Parameter(ckpt["shN"][sampled_pts_idx]), 2.5e-3 / 20))
        else:
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))  

    if cfg.model_type == "textured_gaussians":
        textures = torch.ones(points.shape[0], cfg.texture_resolution, cfg.texture_resolution, 4)
        textures[:, :, :, :3] = 0.1 # init color to low value
        textures[:, :, :, 3] = 1.0 # init alpha to 1.0
        params.append(("textures", torch.nn.Parameter(textures), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

if __name__ == "__main__":
    cfg = Config()
    runner = Runner(cfg)
    
    ckpt = torch.load(cfg.ckpt, map_location=runner.device)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k]
    # runner.render_traj(step=ckpt["step"])
    time.sleep(1000000)