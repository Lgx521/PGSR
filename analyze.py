import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
# import torchvision.utils as vutils

# 从 PGSR 项目中导入必要的模块
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

"""
New version with depth distribution analyze

"""

# K 值应与 C++ 代码和 gaussian_renderer 中的值保持一致
K_BUCKET_SIZE = 64

def visualize_ray_distribution(uid, px, H, W, render_pkg, output_dir):
    """
    可视化指定像素渲染射线上的权重和深度分布。
    """
    px_x, px_y = px
    # 将二维像素坐标转换为一维索引
    pix_idx = px_y * W + px_x
    device = "cuda"

    # --- 从 render_pkg 中提取所有需要的数据 ---
    # !! 重要前提：render函数必须返回 'per_pixel_depths' !!
    if "per_pixel_depths" not in render_pkg:
        print("\n错误: render_pkg 中未找到 'per_pixel_depths'。请修改您的 render 函数以返回深度信息。")
        return

    all_weights = render_pkg["per_pixel_weights"].view(-1, K_BUCKET_SIZE)
    all_depths = render_pkg["per_pixel_depths"].view(-1, K_BUCKET_SIZE)
    all_counts = render_pkg["per_pixel_count"]

    # --- 提取指定像素的数据 ---
    count = int(all_counts[pix_idx].item())
    if count == 0:
        print(f"\n警告: 视角 {uid} 的像素 ({px_x}, {px_y}) 没有高斯函数贡献，无法生成分布图。")
        return

    # 获取有效权重和深度，并转移到 CPU
    weights = all_weights[pix_idx, :count].cpu()
    depths = all_depths[pix_idx, :count].cpu()

    # --- 计算累积 alpha (模拟橙色曲线) ---
    # 权重本质上是 alpha_i * T_{i-1}，累加它们可以得到最终像素的 alpha 值
    accumulated_alpha = torch.cumsum(weights, dim=0)

    # --- 使用 Matplotlib 绘图 ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制权重分布 (使用 stem plot 模拟绿色曲线和标记)
    markerline, stemlines, baseline = ax.stem(
        depths, weights, linefmt='g-', markerfmt='go', basefmt=' '
    )
    plt.setp(stemlines, 'linewidth', 1.5)
    plt.setp(markerline, 'markersize', 4)

    # 绘制累积 Alpha (橙色曲线)
    ax.plot(depths, accumulated_alpha, 'orange', linewidth=2, label='Accumulated Alpha')

    # 美化图表
    ax.set_title(f'Ray Distribution at View {uid}, Pixel ({px_x}, {px_y})')
    ax.set_xlabel('Depth along Ray')
    ax.set_ylabel('Weight / Accumulated Alpha')
    ax.set_ylim(0, 1.05) # Y轴范围从0到1
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(['Per-Gaussian Weight', 'Accumulated Alpha'])

    # 保存图像
    plot_filename = os.path.join(output_dir, f"view_{uid}_pixel_{px_x}_{px_y}_distribution.png")
    plt.savefig(plot_filename)
    plt.close(fig) # 关闭图像以释放内存
    print(f"\n已为视角 {uid} 的像素 ({px_x}, {px_y}) 生成权重分布图: {plot_filename}")


def analyze_scene(args):
    """
    加载模型并使用 GPU 加速计算不确定性热图。
    """
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.skip_train:
        viewpoints = scene.getTestCameras()
        print("正在分析测试集相机...")
    else:
        viewpoints = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
        print("正在分析所有相机...")
        
    if not viewpoints:
        print("未找到可分析的相机视角。")
        return

    output_dir = os.path.join(args.model_path, f"analysis_entropy_K{K_BUCKET_SIZE}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"分析结果将保存到: {output_dir}")

    for viewpoint_cam in tqdm(viewpoints, desc="正在分析视角"):
        H, W = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
        npix = H * W
        device = "cuda"

        # 调用我们修改过的 render 函数，传入 K 值
        render_pkg = render(viewpoint_cam, gaussians, args, background, K=K_BUCKET_SIZE)
        
        # --- 新增：检查是否需要为当前视角生成可视化分布图 ---
        if args.visualize_uid is not None and viewpoint_cam.uid == args.visualize_uid:
            if args.visualize_px is not None:
                visualize_ray_distribution(viewpoint_cam.uid, args.visualize_px, H, W, render_pkg, output_dir)
            else:
                print("\n警告: 已指定 --visualize_uid 但未指定 --visualize_px。跳过分布图生成。")

        # --- GPU 加速计算开始 ---
        counts = render_pkg["per_pixel_count"]
        weights = render_pkg["per_pixel_weights"]
        overflow = render_pkg["per_pixel_overflow"]
        weights = weights.view(npix, K_BUCKET_SIZE)
        arange_k = torch.arange(K_BUCKET_SIZE, device=device)[None, :]
        mask = arange_k < counts.long()[:, None]
        weights = weights * mask
        sums = torch.sum(weights, dim=1, keepdim=True)
        probabilities = weights / (sums + 1e-12)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12) * mask, dim=1)
        entropy_map = entropy.view(H, W)
        # --- GPU 加速计算结束 ---

        overflow_rate = torch.sum(overflow.long() > 0) / npix
        if overflow_rate > 0.05:
            print(f"\n警告: 视角 {viewpoint_cam.uid} 有 {overflow_rate.item():.2%} 的像素溢出 (K={K_BUCKET_SIZE})。")
            
        rendered_image = render_pkg["render"]
        base_name = str(viewpoint_cam.uid)
        entropy_map_np = entropy_map.cpu().numpy()
        normalized_entropy = (entropy_map_np - entropy_map_np.min()) / (entropy_map_np.max() - entropy_map_np.min() + 1e-9)
        heatmap_img = plt.get_cmap('viridis')(normalized_entropy)
        
        vutils.save_image(rendered_image, os.path.join(output_dir, f"{base_name}_render.png"))
        plt.imsave(os.path.join(output_dir, f"{base_name}_entropy_heatmap.png"), heatmap_img)

    print(f"\n分析完成。所有热图已保存至 {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="为已训练的 PGSR 模型生成不确定性分析图。")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int, help="从指定的迭代次数加载模型，-1 代表最新。")
    parser.add_argument("--skip_train", action="store_true", help="如果设置，则仅分析测试集视角。")
    parser.add_argument("--quiet", action="store_true")
    
    # --- 新增命令行参数 ---
    parser.add_argument("--visualize_uid", type=int, default=None, help="指定要可视化射线的视角的 UID。")
    parser.add_argument("--visualize_px", type=int, nargs=2, default=None, metavar=('X', 'Y'),
                        help="指定要可视化射线的像素坐标 (x y)。")

    args = get_combined_args(parser)
    
    if args.iteration == -1:
        point_cloud_dir = os.path.join(args.model_path, "point_cloud")
        try:
            available_iters = sorted([int(f.split("_")[1]) for f in os.listdir(point_cloud_dir) if f.startswith("iteration_")])
            args.iteration = available_iters[-1]
        except (FileNotFoundError, IndexError):
            print(f"错误: 在 {point_cloud_dir} 中找不到已训练的模型检查点。")
            exit(1)

    print(f"从迭代次数 {args.iteration} 加载已训练的模型...")
    safe_state(args.quiet)
    analyze_scene(args)