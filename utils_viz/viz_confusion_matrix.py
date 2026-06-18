import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ======= 混淆矩阵数据 (示例) =======
cm = np.array([
    [126,  13,   2,   0,   3,   0],
    [ 14, 166,   7,   0,   2,   0],
    [  5,   1, 153,   0,  15,   3],
    [  2,   4,   5, 118,   5,   2],
    [  6,   7,   5,   4, 174,   0],
    [  6,   3,   3,  21,   1, 124]
])
# ==========================================

def plot_cm():
    # 计算百分比矩阵
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    labels = ["Agaric", "Broccoli", "Egg Tarts", "Dumplings", "Mapo Tofu", "Purple Buns"]

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    
    # 绘制热力图
    # annot=True 显示数值, fmt='.1f' 保留一位小数
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Accuracy %)')
    plt.tight_layout()
    plt.savefig('./paper_images/fig5_confusion_matrix.png', dpi=300)
    print("已生成: fig5_confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    plot_cm()