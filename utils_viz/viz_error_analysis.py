import cv2
import matplotlib.pyplot as plt
import os

# 确保这两张图在你的硬盘上是存在的
path_dish_5 = r'./my_dataset/zishu/000278.jpg'  # 请修改为真实的 Dish 5 图片路径
path_dish_3 = r'./my_dataset/dumplings/000231.jpg'  # 请修改为真实的 Dish 3 图片路径

# 对应真实的菜名 (用于显示标题)
name_5 = "Purple Buns (True Label)"
name_3 = "Dumplings (Confused Label)"
# ========================================================

def show_comparison():
    # 1. 读取原图
    img5_org = cv2.imread(path_dish_5)
    img3_org = cv2.imread(path_dish_3)

    if img5_org is None or img3_org is None:
        print("错误：找不到图片，请检查 path_dish_5 和 path_dish_3 的路径是否正确！")
        return

    # 2. 转换为 RGB (Matplotlib 显示用)
    img5_org = cv2.cvtColor(img5_org, cv2.COLOR_BGR2RGB)
    img3_org = cv2.cvtColor(img3_org, cv2.COLOR_BGR2RGB)

    # 3. 生成模型视角的 32x32 图
    img5_32 = cv2.resize(img5_org, (32, 32))
    img3_32 = cv2.resize(img3_org, (32, 32))

    # 4. 绘图 (2行2列)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # 第一行：Dish 5
    axes[0, 0].imshow(img5_org)
    axes[0, 0].set_title(f"{name_5}\nOriginal", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img5_32)
    axes[0, 1].set_title(f"{name_5}\nModel View (32x32)", fontsize=12)
    axes[0, 1].axis('off')

    # 第二行：Dish 3
    axes[1, 0].imshow(img3_org)
    axes[1, 0].set_title(f"{name_3}\nOriginal", fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img3_32)
    axes[1, 1].set_title(f"{name_3}\nModel View (32x32)", fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    
    # 保存
    save_path = './paper_images/fig_error_analysis.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 对比图已生成: {save_path}")
    plt.show()

if __name__ == '__main__':
    show_comparison()