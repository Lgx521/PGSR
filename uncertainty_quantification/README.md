好的，这是一个为 `analyze.py` 脚本编写的详细 `README.md` 文件。您可以将此内容保存为 `README.md` 并放置在您的 `PGSR` 项目根目录下。

-----

# Uncertainty Analysis Script (`analyze.py`) Readme

本脚本用于对已训练的 PGSR (Planar-based Gaussian Splatting) 模型进行不确定性分析。它会加载一个训练好的高斯点云模型，并为指定的相机视角生成基于渲染权重分布熵（Entropy of weight distribution）的不确定性热图。

这个实现遵循了 `possible_version.md` 文件中提出的**方案 A (固定桶 K)**。

## 先决条件

在运行此脚本之前，您必须确保已完成以下**所有**准备工作：

1.  **修改并编译 C++ 库**：`diff-plane-rasterization` 子模块的 C++ 和 CUDA 源代码已被修改，以支持在 `forward` 过程中附加输出每个像素的高斯点贡献信息（ID、权重、计数等）。
2.  **成功编译**：修改后的 `diff-plane-rasterization` 库已经**没有错误地**成功编译和安装到您的 Python 环境中。
3.  **更新 Python 封装**：`diff-plane-rasterization/__init__.py` 文件已被更新，以正确处理与修改后的 C++ 库之间的参数传递。
4.  **更新渲染器**：`gaussian_renderer/__init__.py` 文件中的 `render` 函数已被更新，能够创建并传递用于收集不确定性数据的缓冲区，并能接收 `K` 值作为参数。

## 使用方法

在终端中，从 `PGSR` 项目的根目录运行脚本。

### 命令格式

```bash
python analyze.py -m <model_path> -s <source_path> [可选参数]
```

### 参数说明

  * `-m`, `--model_path` **(必需)**

      * **说明**: 已训练模型的输出主目录。这应该是您在训练时为 `-m` 参数指定的同一个路径。脚本会从此目录加载模型配置、相机参数和最新的点云检查点。
      * **示例**: `./data/all_output/imgs_4_test`

  * `-s`, `--source_path` **(必需)**

      * **说明**: 用于训练该模型的原始数据集所在的文件夹路径。脚本需要此路径来加载场景信息。
      * **示例**: `./data/antique_gt_pose/scene_4`

  * `--iteration` *(可选)*

      * **说明**: 指定加载模型的迭代次数。如果未提供，脚本将自动加载训练目录中最新的一个检查点。
      * **示例**: `--iteration 7000`

  * `--skip_train` *(可选)*

      * **说明**: 如果包含此标志，脚本将只分析被划分为“测试集”的相机视角。如果省略，脚本将分析数据集中所有的相机（训练集 + 测试集）。
      * **注意**: 如果您的数据集很小（例如少于8张图），可能没有相机被默认规则划分为测试集，此时使用此标志将不会生成任何结果。

### 关键内部参数

  * `K_BUCKET_SIZE`
      * **位置**: `analyze.py` 脚本内部。
      * **说明**: 定义了每个像素最多可以记录多少个高斯点的贡献信息。此值应与 C++ 代码中的预期相匹配。如果分析过程中出现大量像素“溢出”的警告，您可能需要增大此值，并相应地修改 C++ 代码后重新编译。
      * **默认值**: `64`

## 使用实例

### 命令

```bash
python analyze.py -m ./data/all_output/imgs_4_test -s ./data/antique_gt_pose/scene_4
```

### 命令解析

  * `python analyze.py`: 执行分析脚本。
  * `-m ./data/all_output/imgs_4_test`: 指定加载位于 `./data/all_output/imgs_4_test` 目录下的训练成果。
  * `-s ./data/antique_gt_pose/scene_4`: 告诉脚本该模型是使用 `./data/antique_gt_pose/scene_4` 目录下的原始数据训练的。
  * *(此示例中未包含 `--skip_train`)*: 因此，脚本将分析并为该场景中的所有4个相机视角生成热图。

## 输出结果

脚本运行后，会在您的模型目录 (`-m` 指定的路径)下创建一个新的子目录，例如：

`./data/all_output/imgs_4_test/analysis_entropy_K64/`

在该目录中，会为每个分析过的相机视角生成两张图片，以相机的唯一ID命名：

1.  **`[相机UID]_render.png`**

      * PGSR 模型在该视角下的原始渲染图像，用于对比。

2.  **`[相机UID]_entropy_heatmap.png`**

      * **不确定性热图**。这是您最终想要的结果。
      * **如何解读**:
          * **暗色/紫色区域**: 代表低熵值，表示该像素的渲染由少数几个高斯点主导，重建质量较高，**不确定性低**。
          * **亮色/黄色区域**: 代表高熵值，表示该像素的渲染由大量高斯点微弱地贡献而成，重建质量可能较差，**不确定性高**。

## 注意事项

  * **内存消耗**: 此脚本（尤其是在渲染过程中创建的缓冲区）可能会占用大量的 GPU 显存。如果遇到显存不足的错误，可以考虑减小 `K_BUCKET_SIZE` 的值（需同步修改C++代码并重新编译）。
  * **像素溢出**: 如果在运行时看到关于“像素溢出”的警告，这意味着在某些像素上，有超过 `K_BUCKET_SIZE` 个高斯点对其有贡献。少量溢出通常不影响整体分析，但如果溢出率过高，则得到的热图可能不完全准确。


  ---
  # 特定像素深度渲染可视化

  ```bash
  python analyze.py -m ./data/all_output/imgs_4_test -s ./data/antique_gt_pose/scene_4 --visualize_uid 0 --visualize_px 400 400
  ```
