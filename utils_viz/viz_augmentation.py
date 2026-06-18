import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# 配置
IMG_PATH = './my_dataset/mapoTofu/000214.jpg'  # 请替换为你本地任意一张存在的图片路径
IMG_SIZE = 32

def show_augmentation_grid():
    # 读取并预处理原图
    original = cv2.imread(IMG_PATH)
    if original is None:
        print(f"找不到图片: {IMG_PATH}，请修改代码中的路径")
        return
    original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) # Matplotlib 需要 RGB

    aug_images = [("Original", original)]

    # 1. 旋转 (Rotation)
    rows, cols, _ = original.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1) # 旋转30度演示
    rot = cv2.warpAffine(original, M, (cols, rows))
    aug_images.append(("Rotation", rot))

    # 2. 翻转 (Flip)
    flip = cv2.flip(original, 1)
    aug_images.append(("Flip", flip))

    # 3. 亮度 (Brightness)
    bright = cv2.add(original, np.array([50.0])) # 变亮
    aug_images.append(("Bright", bright))

    # 4. 挖孔 (Cutout - Failed Case)
    cutout = original.copy()
    mask_size = 10
    cutout[0:mask_size, 0:mask_size, :] = 0 # 左上角挖黑
    aug_images.append(("Cutout\n(Failed)", cutout))

    # 绘图
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, (title, img) in enumerate(aug_images):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('./paper_images/fig2_augmentation.png', dpi=300)
    print("已生成: fig2_augmentation.png")
    plt.show()

if __name__ == '__main__':
    show_augmentation_grid()