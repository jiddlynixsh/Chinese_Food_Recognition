import numpy as np
import matplotlib.pyplot as plt
import load
import cv2

def plot_mean_images():
    print("正在生成最终版图片...")
    samples, labels = load._train_samples, load._train_labels
    samples = (samples + 1.0) * 128.0
    samples = np.clip(samples, 0, 255).astype(np.uint8)
    label_indices = np.argmax(labels, axis=1)
    
    # 类别名称
    class_names = [
        "Agaric", "Broccoli", "Egg Tarts", 
        "Dumplings", "Mapo Tofu", "Purple Buns"
    ]
    
    mean_images = []
    for i in range(6):
        imgs = samples[label_indices == i]
        if len(imgs) == 0:
            mean_img = np.zeros((32,32,3), dtype=np.uint8)
        else:
            mean_img = np.mean(imgs, axis=0).astype(np.uint8)
        
        # BGR -> RGB 修正
        mean_img = cv2.cvtColor(mean_img, cv2.COLOR_BGR2RGB)
        mean_images.append(mean_img)

    # === 排版调整 ===
    # figsize: 宽度18，高度4 (增加高度，给标题留空间)
    fig, axes = plt.subplots(1, 6, figsize=(18, 4)) 
    
    for i, img in enumerate(mean_images):
        axes[i].imshow(img)
        # pad=20: 让子标题离图片远一点
        axes[i].set_title(class_names[i], fontsize=14, pad=15) 
        axes[i].axis('off')
        
    # y=1.05: 把大标题往上提，提到画板边缘上面去一点
    plt.suptitle("Average Color Representation per Class", fontsize=20, y=1.02, fontweight='bold')
    
    # tight_layout: 自动调整，不让子图重叠
    plt.tight_layout()
    
    save_path = './paper_images/fig3_mean_color_final.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches='tight' 保证不切掉大标题
    print(f"最终完美版已生成: {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_mean_images()