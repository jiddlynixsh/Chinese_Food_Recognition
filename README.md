# Chinese_Food_Recognition
A Lightweight Chinese Cuisine Recognition System Based on Tensorflow

## ♥ 项目亮点 Key Features
- **轻量化设计 Lightweight Architecture**：专为边缘设备优化，模型参数量小，推理速度快。
- **可解释性分析 Model Explainability**：利用 Grad-CAM 进行了特征图可视化，直观展示模型对低分辨率菜品的核心关注区域。
- **丰富的数据分析 Comprehensive Analysis**：包含混淆矩阵分析、错误分析以及数据增强可视化脚本。

## 📚 实验结果 Results & Visualization
### 特征图可视化 Grad-CAM
这里展示你的模型聚焦在哪里，比如通过热力图看它是不是真的在看“菜”，而不是看“盘子”：
![Grad-CAM Result](docs/paper_images/gradcam_result.png) 

### 👍性能指标 Metrics
- **准确率 Accuracy**: 86.1% (在低分辨率/特定数据集上)

## 🛠️ 环境依赖 Requirements
- Python 3.x
- TensorFlow 1.x ⚠️ (请务必使用 1.x 版本，本项目基于 TensorFlow 1.x 架构开发)
- OpenCV, Matplotlib, NumPy

## 🚀 快速开始 Usage
1. 下载本项目并安装依赖。
2. 将数据集（`train.mat` 和 `test.mat`）放入项目根目录。
3. 运行主程序启动训练与评估：
   ```bash

## 💾 数据集准备 (Dataset Preparation)

[cite_start]本项目使用的是 **ChineseFoodNet** 数据集的精简子集，包含 6 类常见中国菜品（木耳、西兰花、蛋挞、水饺、麻婆豆腐、紫薯包） 。

### 1. 核心数据文件
仓库中已自带预处理好的 `train.mat` 和 `test.mat`（约 3MB 左右），无需额外下载。

### 2. 真实图片数据（用于运行可视化脚本）
由于图片文件较大，未直接上传至 GitHub 代码库。
- **完整图片数据集下载链接**：[👉 点击这里下载数据集](https://sites.google.com/view/chinesefoodnet/)
- **下载后配置**：请将下载好的图片解压，并确保图片存放在根目录的 `my_dataset/` 文件夹下。

### 📁 推荐运行目录结构
[cite_start]为了确保可视化脚本（如 Grad-CAM [cite: 329]）能够顺利读取真实图片，请保证你的本地目录结构如下：
```text
├── my_dataset/            # 💡 请将下载的真实食物图片文件夹放在这里
│   ├── Agaric/
│   ├── Broccoli/
│   └── ...
├── train.mat              # 已自带
├── test.mat               # 已自带
├── src/
├── utils_viz/             # 运行此文件夹下的脚本需要 my_dataset 中的图片
├── main.py
└── README.md
