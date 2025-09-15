# analyze.py

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

# --- 我们在 C++ 中实现的方案A的参数 ---
# K 值定义了每个像素最多可以记录多少个高斯点的贡献。
# 如果您在 C++ 代码中修改了这个值，请在此处保持同步。
K_BUCKET_SIZE = 64

def analyze_scene(args):
    """
    加载已训练的模型，遍历相机视角，并为每个视角计算和保存不确定性热图。
    """
    # 加载训练好的模型和场景数据
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 选择要分析的相机视角
    if args.skip_train:
        viewpoints = scene.getTestCameras()
        print("正在分析测试集相机...")
    else:
        viewpoints = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
        print("正在分析所有相机...")
        
    if not viewpoints:
        print("未找到可分析的相机视角。如果使用了 --skip_train，请确保您的数据集包含测试集。")
        return

    # 创建输出目录
    output_dir = os.path.join(args.model_path, f"analysis_entropy_K{K_BUCKET_SIZE}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"分析结果将保存到: {output_dir}")

    # 主分析循环
    for viewpoint_cam in tqdm(viewpoints, desc="正在分析视角"):
        H, W = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
        
        # 调用我们修改过的 render 函数，传入 K 值
        render_pkg = render(viewpoint_cam, gaussians, args, background, K=K_BUCKET_SIZE)
        
        # 从返回的字典中提取我们新增的缓冲区
        counts = render_pkg["per_pixel_count"].cpu().numpy()
        weights = render_pkg["per_pixel_weights"].cpu().numpy()
        overflow = render_pkg["per_pixel_overflow"].cpu().numpy()

        npix = H * W
        weights = weights.reshape(npix, K_BUCKET_SIZE)
        
        # 检查并报告溢出率
        overflow_rate = np.sum(overflow > 0) / npix
        if overflow_rate > 0.05: # 如果超过5%的像素溢出
            print(f"\n警告: 视角 {viewpoint_cam.uid} 有 {overflow_rate:.2%} 的像素溢出 (K={K_BUCKET_SIZE})。为获得更精确结果，可考虑在 C++ 代码中增大 K 值并重新编译。")
            
        # 核心：计算每个像素的熵
        entropy_list = []
        for i in range(npix):
            n = int(min(counts[i], K_BUCKET_SIZE))
            if n <= 1: # 如果只有一个或没有贡献，熵为0
                entropy_list.append(0.0)
                continue
            
            w = weights[i, :n]
            
            # 归一化为概率分布
            s = np.sum(w)
            if s <= 1e-8: # 避免除以零
                entropy_list.append(0.0)
                continue
            
            p = w / s
            # 过滤掉概率极小的值以保证数值稳定性
            p = p[p > 1e-12]
            
            # 计算熵 H(p) = - sum(p * log(p))
            entropy = -np.sum(p * np.log(p))
            entropy_list.append(entropy)
            
        # 将熵列表重塑为图像
        entropy_map = np.array(entropy_list).reshape(H, W)
        
        # 可视化并保存结果
        rendered_image = render_pkg["render"]
        base_name = str(viewpoint_cam.uid) # 使用唯一的相机ID作为文件名

        # 归一化熵图以便于可视化
        normalized_entropy = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-9)
        
        # 使用 'inferno' 色彩映射表生成热图
        heatmap_img = plt.get_cmap('inferno')(normalized_entropy)
        
        # 保存渲染图和热图
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
    
    # 自动查找最新的迭代次数
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