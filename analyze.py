import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# 从 PGSR 项目中导入必要的模块
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

# K 值应与 C++ 代码和 gaussian_renderer 中的值保持一致
K_BUCKET_SIZE = 64

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
        
        # --- GPU 加速计算开始 ---
        # 1. 直接在 GPU 上获取数据
        counts = render_pkg["per_pixel_count"]        # shape: [npix], on GPU
        weights = render_pkg["per_pixel_weights"]      # shape: [npix * K], on GPU
        overflow = render_pkg["per_pixel_overflow"]    # shape: [npix], on GPU

        weights = weights.view(npix, K_BUCKET_SIZE) # Reshape

        # 2. 创建一个 mask 来处理每个像素不同的贡献数量
        # arange_k shape: [K] -> [1, K]
        arange_k = torch.arange(K_BUCKET_SIZE, device=device)[None, :]
        # counts shape: [npix] -> [npix, 1]
        # mask shape: [npix, K]
        mask = arange_k < counts.long()[:, None]

        # 3. 应用 mask 并归一化权重
        weights = weights * mask
        sums = torch.sum(weights, dim=1, keepdim=True)
        # 添加一个极小值 eps 以防止除以零
        probabilities = weights / (sums + 1e-12)

        # 4. 计算熵 H(p) = - sum(p * log(p))
        # 再次乘以 mask 是为了确保 p=0 的位置不会因为 log(eps) 产生非零熵
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12) * mask, dim=1)
        
        # 5. 将熵张量重塑为图像
        entropy_map = entropy.view(H, W)
        # --- GPU 加速计算结束 ---

        # 检查并报告溢出率
        overflow_rate = torch.sum(overflow.long() > 0) / npix
        if overflow_rate > 0.05:
            print(f"\n警告: 视角 {viewpoint_cam.uid} 有 {overflow_rate.item():.2%} 的像素溢出 (K={K_BUCKET_SIZE})。")
            
        # 可视化并保存结果
        rendered_image = render_pkg["render"]
        base_name = str(viewpoint_cam.uid)

        # 仅在最后一步将熵图移至 CPU 进行保存
        entropy_map_np = entropy_map.cpu().numpy()
        normalized_entropy = (entropy_map_np - entropy_map_np.min()) / (entropy_map_np.max() - entropy_map_np.min() + 1e-9)
        
        heatmap_img = plt.get_cmap('viridis')(normalized_entropy) # <--- 在这里修改 colormap
        
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