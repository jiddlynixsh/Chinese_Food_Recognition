import tensorflow as tf
import numpy as np
from dp_refined import Network

# 定义和 main.py 一样的参数
image_size = 32
num_channels = 3
num_labels = 6

def calculate_metrics():
    # 重置图
    tf.reset_default_graph()
    
    # 实例化网络 (只用于定义图，不需要加载权重也能算结构)
    # 注意：这里 batch_size 设为 1，用于模拟单张图片推理
    net = Network(
        train_batch_size=1, test_batch_size=1, pooling_scale=2,
        dropout_rate=0.5, base_learning_rate=0.0005, decay_rate=0.99
    )
    
    # 定义输入
    net.define_inputs(
        train_samples_shape=(1, image_size, image_size, num_channels),
        train_labels_shape=(1, num_labels),
        test_samples_shape=(1, image_size, image_size, num_channels),
    )

    # === 【关键修复】这里补上了网络层的定义 ===
    # 必须和 main.py 里的结构完全一致，才能算出正确的 FLOPs
    net.add_conv(patch_size=3, in_depth=num_channels, out_depth=32, activation='relu', pooling=False, name='conv1')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv2')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=False, name='conv3')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv4')

    # 4 = 两次 pooling, 每一次缩小为 1/2 (32 -> 16 -> 8)
    # 32 = conv4 out_depth
    net.add_fc(in_num_nodes=(image_size // 4) * (image_size // 4) * 32, out_num_nodes=256, activation='relu',
               name='fc1')
    net.add_fc(in_num_nodes=256, out_num_nodes=6, activation=None, name='fc2')
    # ==========================================
    
    # 构建模型
    net.define_model()
    
    print("-" * 30)
    print("开始计算指标...")

    # === 1. 计算参数量 (Parameters) ===
    # 统计所有 trainable variables 的大小
    total_parameters = 0
    for variable in tf.trainable_variables():
        # variable.shape 是一个 tuple, np.prod 相乘得到元素个数
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    
    print(f"Total Parameters: {total_parameters}")
    print(f"Model Size (approx): {total_parameters * 4 / 1024 / 1024:.2f} MB") # float32 = 4 bytes
    
    # === 2. 计算 FLOPs (浮点运算量) ===
    # TensorFlow 1.x 自带 profiler 可以算
    try:
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(graph=tf.get_default_graph(),
                                    run_meta=run_meta, cmd='op', options=opts)
        
        if flops is not None:
            print(f"Total FLOPs: {flops.total_float_ops}")
            print(f"MFLOPs (Million FLOPs): {flops.total_float_ops / 1e6:.2f}")
    except Exception as e:
        print(f"FLOPs 计算出错 (可能是 TF 版本兼容问题): {e}")

    # === 3. 估算 CPU 推理时间 (Latency) ===
    print("正在测试推理延迟 (运行 100 次)...")
    # 跑 100 次取平均
    import time
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # 随机生成一张图
    dummy_input = np.random.rand(1, 32, 32, 3)
    
    # 预热
    for _ in range(10):
        # 注意：这里要用 single_prediction 或者 test_prediction，
        # 因为我们在 define_model 里通过 train=False 分支定义了测试图
        # 但最简单的是直接跑 net.test_prediction
        sess.run(net.test_prediction, feed_dict={net.tf_test_samples: dummy_input})
        
    start_time = time.time()
    for _ in range(100):
        sess.run(net.test_prediction, feed_dict={net.tf_test_samples: dummy_input})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Average Inference Time (CPU): {avg_time * 1000:.2f} ms")
    sess.close()

if __name__ == '__main__':
    calculate_metrics()