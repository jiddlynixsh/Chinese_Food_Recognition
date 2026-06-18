import visualkeras
from tensorflow.keras import layers, models
from collections import defaultdict

# ==========================================
# 1. 这里复现网络结构 (只为了画图，不训练)
# ==========================================
model = models.Sequential()

# --- 输入层 (32x32) ---
model.add(layers.InputLayer(input_shape=(32, 32, 3)))

# --- Block 1 (对应 main.py 的 conv1 和 conv2) ---
# conv1: 32核, 无池化
model.add(layers.Conv2D(32, (3, 3), padding='same', name='Conv1')) 
model.add(layers.BatchNormalization(name='BN1'))
model.add(layers.LeakyReLU(alpha=0.1, name='LReLU1'))

# conv2: 32核, 有池化
model.add(layers.Conv2D(32, (3, 3), padding='same', name='Conv2')) 
model.add(layers.BatchNormalization(name='BN2'))
model.add(layers.LeakyReLU(alpha=0.1, name='LReLU2'))
model.add(layers.MaxPooling2D((2, 2), name='Pool1')) # 图片变 16x16

# --- Block 2 (对应 main.py 的 conv3 和 conv4) ---
# conv3: 32核, 无池化
model.add(layers.Conv2D(32, (3, 3), padding='same', name='Conv3'))
model.add(layers.BatchNormalization(name='BN3'))
model.add(layers.LeakyReLU(alpha=0.1, name='LReLU3'))

# conv4: 32核, 有池化
model.add(layers.Conv2D(32, (3, 3), padding='same', name='Conv4'))
model.add(layers.BatchNormalization(name='BN4'))
model.add(layers.LeakyReLU(alpha=0.1, name='LReLU4'))
model.add(layers.MaxPooling2D((2, 2), name='Pool2')) # 图片变 8x8

# --- 分类头 (Fully Connected) ---
model.add(layers.Flatten(name='Flatten'))
model.add(layers.Dense(256, name='FC1'))
model.add(layers.Dropout(0.5, name='Dropout'))
model.add(layers.Dense(6, activation='softmax', name='Output'))

# ==========================================
# 2. 配色与生成 (这是魔法发生的时刻)
# ==========================================
# 定义每种层的颜色 
color_map = defaultdict(dict)
color_map[layers.Conv2D]['fill'] = '#FFD166'      # 卷积层：黄色
color_map[layers.BatchNormalization]['fill'] = '#06D6A0' # BN层：绿色
color_map[layers.LeakyReLU]['fill'] = '#118AB2'   # 激活层：蓝色
color_map[layers.MaxPooling2D]['fill'] = '#EF476F' # 池化层：红色
color_map[layers.Dense]['fill'] = '#073B4C'       # 全连接：深蓝
color_map[layers.Flatten]['fill'] = '#EEE'        # 打平：灰色

# 生成图片
# spacing: 层之间的间距
# scale_xy: 图片大小缩放
# draw_volume: 是否画成立体块
font_path = "arial.ttf" # 如果报错找不到字体，可以删掉 font 参数

print("正在生成架构图，请稍候...")
visualkeras.layered_view(model, 
                         legend=True,          # 显示图例
                         draw_volume=True,     # 画立体
                         spacing=20,           # 拉开间距防止太挤
                         color_map=color_map,  # 使用上面的配色
                         scale_xy=3,           # 放大一点
                         to_file='my_architecture_3d.png').show()

print("成功！图片已保存为 my_architecture_3d.png，快打开看看！")