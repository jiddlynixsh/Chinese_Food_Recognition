import os
import cv2
import numpy as np
import random
import scipy.io as sio  # 用于保存 mat 文件

# 我的图片文件夹路径
data_dir = r'./my_dataset' 

# === 1. [核心修改] 改回 32x32 ===
IMG_SIZE = 32

# 扩充倍数
AUGMENT_FACTOR = 5

def random_augment(img):
    """
    随机生成一张增强后的图片
    包含：旋转、翻转、亮度、以及【新增】Cutout挖孔
    """
    rows, cols, _ = img.shape
    new_img = img.copy()

    # 1. 随机旋转 
    angle = random.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    new_img = cv2.warpAffine(new_img, M, (cols, rows))

    # 2. 随机翻转 (50% 概率)
    if random.random() > 0.5:
        new_img = cv2.flip(new_img, 1)

    # 3. 随机亮度调整
    brightness = random.uniform(-30, 30)
    if brightness > 0:
        new_img = cv2.add(new_img, np.array([brightness]))
    else:
        new_img = cv2.subtract(new_img, np.array([abs(brightness)]))
    
    # # === 4. [新增] Cutout (挖孔) ===
    # # 在 32x32 的图上，挖掉 8 到 16 像素的方块
    # # 这会遮挡很大一部分信息，复现论文中提到的"破坏拓扑结构"
    # mask_size = random.randint(8, 16) 
    
    # # 随机中心点
    # y_center = random.randint(0, rows)
    # x_center = random.randint(0, cols)
    
    # # 计算边界并截断，防止越界
    # y1 = np.clip(y_center - mask_size // 2, 0, rows)
    # y2 = np.clip(y_center + mask_size // 2, 0, rows)
    # x1 = np.clip(x_center - mask_size // 2, 0, cols)
    # x2 = np.clip(x_center + mask_size // 2, 0, cols)
    
    # # 填黑 (0)
    # new_img[y1:y2, x1:x2, :] = 0 

    return new_img

def create_mat_files():
    # 建立两个列表，分别存放训练集和测试集
    train_list = []
    test_list = []
    
    categories = os.listdir(data_dir) # 扫描有哪些文件夹
    print(f"检测到分类: {categories}")
    
    # 1. 读取并扩充数据
    for class_num, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        print(f"正在处理: {category} (Label: {class_num})...")
        
        # 先临时存放这一类的所有原图
        current_class_imgs = []
        
        # 遍历文件夹里的每一张照片
        for img_name in os.listdir(path):
            try:
                # 读取照片
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                if img_array is None: continue # 坏了就跳过
                # 调整缩放为 IMG_SIZE (32)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                current_class_imgs.append(new_array)
            except Exception:
                pass

        # 在原图层面先切分，防止测试集污染
        random.shuffle(current_class_imgs)
        
        # 按照 4:1 划分训练集和测试集 (80% 训练, 20% 测试)
        split_index = int(len(current_class_imgs) * 0.8)
        train_org = current_class_imgs[:split_index] # 这一份用于训练
        test_org = current_class_imgs[split_index:]  # 这一份用于测试

        # --- 处理训练集 (Train) ---
        for img in train_org:
            # 放入原图
            train_list.append([img, class_num])
            # 每张图生成 AUGMENT_FACTOR 张随机图
            for _ in range(AUGMENT_FACTOR):
                aug_img = random_augment(img)
                train_list.append([aug_img, class_num])
        
        # --- 处理测试集 (Test) ---
        for img in test_org:
            # 测试集只放原图
            test_list.append([img, class_num])

    print(f"处理完成")

    # 2. 打乱数据顺序
    random.shuffle(train_list)
    random.shuffle(test_list)

    print(f"训练集数量: {len(train_list)}")
    print(f"测试集数量: {len(test_list)}")

    # 3. 转换格式并保存
    save_to_mat(train_list, 'train.mat')
    save_to_mat(test_list, 'test.mat')

def save_to_mat(dataset, filename):
    X = [] # 存放像素
    y = [] # 存放标签
    for features, label in dataset:
        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)

    # 适配 load.py 的输入格式 (N, H, W, C) -> (H, W, C, N)
    # 这里的 transpose 是为了匹配 load.py 里的 def reformat
    X = X.transpose(1, 2, 3, 0)
    y = y.reshape(-1, 1)

    sio.savemat(filename, {'X': X, 'y': y})
    print(f"已保存: {filename} | X shape: {X.shape}")

if __name__ == '__main__':
    create_mat_files()