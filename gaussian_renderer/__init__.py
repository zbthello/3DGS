import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    渲染场景。
    参数:
        viewpoint_camera - 摄像机视角
        pc - 高斯模型
        pipe - 管道参数
        bg_color - 背景颜色张量，必须在GPU上
        scaling_modifier - 缩放修饰符，默认为1.0
        override_color - 覆盖颜色，默认为None
    """

    # 创建一个全零张量，用于使PyTorch返回2D（屏幕空间）均值的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # retain_grad()方法的作用是保留特定张量的梯度，以便后续的计算或分析
        screenspace_points.retain_grad()  # 保留梯度信息
    except:
        pass

    # 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 计算视角的X轴正切
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 计算视角的Y轴正切

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 图像高度
        image_width=int(viewpoint_camera.image_width),  # 图像宽度
        tanfovx=tanfovx,  # 视角X轴正切
        tanfovy=tanfovy,  # 视角Y轴正切
        bg=bg_color,  # 背景颜色
        scale_modifier=scaling_modifier,  # 缩放修饰符
        viewmatrix=viewpoint_camera.world_view_transform,  # 世界视图变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform,  # 投影变换矩阵
        sh_degree=pc.active_sh_degree,  # 球谐函数度数
        campos=viewpoint_camera.camera_center,   # 摄像机中心
        prefiltered=False,  # 预过滤
        debug=pipe.debug  # 调试模式
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
