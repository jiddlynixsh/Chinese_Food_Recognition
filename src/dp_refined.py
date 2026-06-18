# 新的 refined api 不支持 Python2
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2                 # <--- 新增
import matplotlib.pyplot as plt # <--- 新增

class Network():
    def __init__(self, train_batch_size, test_batch_size, pooling_scale,
                 dropout_rate, base_learning_rate, decay_rate,
                 optimizeMethod='adam', save_path='model/default.ckpt'):
        """
        @num_hidden: 隐藏层的节点数量
        @batch_size：因为我们要节省内存，所以分批处理数据。每一批的数据量。
        """
        self.optimizeMethod = optimizeMethod
        self.dropout_rate = dropout_rate
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Hyper Parameters
        self.conv_config = []  # list of dict
        self.fc_config = []  # list of dict
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weights = []
        self.fc_biases = []
        self.pooling_scale = pooling_scale
        self.pooling_stride = pooling_scale

        # Graph Related
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None

        # 统计
        self.writer = None
        self.merged = None
        self.train_summaries = []
        self.test_summaries = []

        # save 保存训练的模型
        self.saver = None
        self.save_path = save_path

    def add_conv(self, *, patch_size, in_depth, out_depth, activation='relu', pooling=False, name):
        """
        This function does not define operations in the graph, but only store config in self.conv_layer_config
        """
        self.conv_config.append({
            'patch_size': patch_size,
            'in_depth': in_depth,
            'out_depth': out_depth,
            'activation': activation,
            'pooling': pooling,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(
                tf.truncated_normal([patch_size, patch_size, in_depth, out_depth], stddev=0.1), name=name + '_weights')
            biases = tf.Variable(tf.constant(0.1, shape=[out_depth]), name=name + '_biases')
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)

    def add_fc(self, *, in_num_nodes, out_num_nodes, activation='relu', name):
        """
        add fc layer config to slef.fc_layer_config
        """
        self.fc_config.append({
            'in_num_nodes': in_num_nodes,
            'out_num_nodes': out_num_nodes,
            'activation': activation,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes]))
            self.fc_weights.append(weights)
            self.fc_biases.append(biases)
            self.train_summaries.append(tf.summary.histogram(str(len(self.fc_weights)) + '_weights', weights))
            self.train_summaries.append(tf.summary.histogram(str(len(self.fc_biases)) + '_biases', biases))

    def apply_regularization(self, _lambda):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        # 1e5
        return _lambda * regularization

    # should make the definition as an exposed API, instead of implemented in the function
    def define_inputs(self, *, train_samples_shape, train_labels_shape, test_samples_shape):
        # 这里只是定义图谱中的各种变量
        with tf.name_scope('inputs'):
            self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
            self.tf_train_labels = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_test_samples = tf.placeholder(tf.float32, shape=test_samples_shape, name='tf_test_samples')

    def define_model(self):
        """
        定义我的计算图谱 (自动适配任意分辨率版)
        """

        def model(data_flow, train=True):
            """
            @data: original inputs
            @return: logits
            """
            # === [关键修复] 添加 reuse=tf.AUTO_REUSE，允许测试时复用训练时创建的 BN 参数 ===
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                # Define Convolutional Layers
                for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
                    with tf.name_scope(config['name'] + '_model'):
                        with tf.name_scope('convolution'):
                            # 1. 卷积计算
                            data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
                            data_flow = data_flow + biases
                            
                            # 2. Batch Normalization
                            data_flow = tf.layers.batch_normalization(data_flow, training=train, name=config['name']+'_bn')

                            if not train:
                                # [自动获取尺寸] 不再写死 32，而是动态读取当前层的高度
                                current_size = data_flow.get_shape().as_list()[1]
                                self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                          display_size=current_size, name=config['name'] + '_conv')
                        
                        # 3. 激活函数 (Leaky ReLU)
                        if config['activation'] == 'relu':
                            data_flow = tf.nn.leaky_relu(data_flow, alpha=0.1, name=config['name']+'_leaky_relu')
                            
                            if not train:
                                current_size = data_flow.get_shape().as_list()[1]
                                self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                          display_size=current_size, name=config['name'] + '_relu')
                        else:
                            raise Exception('Activation Func can only be Relu right now. You passed', config['activation'])
                        if config['pooling']:
                            data_flow = tf.nn.max_pool(
                                data_flow,
                                ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                                strides=[1, self.pooling_stride, self.pooling_stride, 1],
                                padding='SAME')
                            if not train:
                                current_size = data_flow.get_shape().as_list()[1]
                                self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                          display_size=current_size, 
                                                          name=config['name'] + '_pooling')
                # 保存最后一层卷积的特征图 (feature maps)，用于 Grad-CAM
                # 只有在处理 single_input (测试阶段) 时，这个变量最后会被保留为 single_input 的图
                if not train:
                    self.last_conv_output = data_flow 
                # ========================

                # Define Fully Connected Layers
                for i, (weights, biases, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):
                    if i == 0:
                        shape = data_flow.get_shape().as_list()
                        data_flow = tf.reshape(data_flow, [shape[0], shape[1] * shape[2] * shape[3]])
                    with tf.name_scope(config['name'] + 'model'):

                        ### Dropout
                        if train and i == len(self.fc_weights) - 1:
                            data_flow = tf.nn.dropout(data_flow, self.dropout_rate, seed=4926)
                        ###

                        data_flow = tf.matmul(data_flow, weights) + biases
                        if config['activation'] == 'relu':
                            data_flow = tf.nn.relu(data_flow)
                        elif config['activation'] is None:
                            pass
                        else:
                            raise Exception('Activation Func can only be Relu or None right now. You passed',
                                            config['activation'])
                return data_flow

        # Training computation.
        logits = model(self.tf_train_samples)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.tf_train_labels))
            self.loss += self.apply_regularization(_lambda=1e-3)
            self.train_summaries.append(tf.summary.scalar('Loss', self.loss))

        # learning rate decay
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.base_learning_rate,
            global_step=global_step * self.train_batch_size,
            decay_steps=100,
            decay_rate=self.decay_rate,
            staircase=True
        )

        # Optimizer.
        with tf.name_scope('optimizer'):
            # === 显式更新 BN 参数 ===
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if (self.optimizeMethod == 'gradient'):
                    self.optimizer = tf.train \
                        .GradientDescentOptimizer(learning_rate) \
                        .minimize(self.loss)
                elif (self.optimizeMethod == 'momentum'):
                    self.optimizer = tf.train \
                        .MomentumOptimizer(learning_rate, 0.5) \
                        .minimize(self.loss)
                elif (self.optimizeMethod == 'adam'):
                    self.optimizer = tf.train \
                        .AdamOptimizer(learning_rate) \
                        .minimize(self.loss)

        # Predictions for the training, validation, and test data.
        with tf.name_scope('train'):
            self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
            tf.add_to_collection("prediction", self.train_prediction)
        with tf.name_scope('test'):
            self.test_prediction = tf.nn.softmax(model(self.tf_test_samples, train=False), name='test_prediction')
            tf.add_to_collection("prediction", self.test_prediction)

            # 单张图片预测的 placeholder 也要动态适配
            current_shape = self.tf_test_samples.get_shape().as_list() # [500, H, W, 3]
            single_shape = (1, current_shape[1], current_shape[2], current_shape[3])
            single_input = tf.placeholder(tf.float32, shape=single_shape, name='single_input')
            self.single_logits = model(single_input, train=False) # 先拿到 Logits（新增）
            self.single_prediction = tf.nn.softmax(self.single_logits, name='single_prediction') # 再做 Softmax
            tf.add_to_collection("prediction", self.single_prediction)

           
        self.merged_train_summary = tf.summary.merge(self.train_summaries)
        self.merged_test_summary = tf.summary.merge(self.test_summaries)

        # 放在定义Graph之后，保存这张计算图
        self.saver = tf.train.Saver(tf.global_variables())

    def run(self, train_samples, train_labels, test_samples, test_labels, *, train_data_iterator, iteration_steps,
            test_data_iterator):
        """
        用到Session
        :data_iterator: a function that yields chuck of data
        """
        self.writer = tf.summary.FileWriter('./board', tf.get_default_graph())

        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.global_variables_initializer().run()

            ### 训练
            print('Start Training')
            # batch 1000
            for i, samples, labels in train_data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                          chunkSize=self.train_batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
                )
                self.writer.add_summary(summary, i)
                # labels is True Labels
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)
            ###

            ### 测试
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in test_data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples}
                )
                self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.print_confusion_matrix(np.add.reduce(confusionMatrices))
            ###

    def train(self, train_samples, train_labels, *, data_iterator, iteration_steps):
        self.writer = tf.summary.FileWriter('./board', tf.get_default_graph())
        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.global_variables_initializer().run()

            ### 训练
            print('Start Training')
            # batch 1000
            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                    chunkSize=self.train_batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
                )
                self.writer.add_summary(summary, i)
                # labels is True Labels
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)
            ###

            # 检查要存放的路径值否存在。这里假定只有一层路径。
            import os
            if os.path.isdir(self.save_path.split('/')[0]):
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)
            else:
                os.makedirs(self.save_path.split('/')[0])
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)

    def test(self, test_samples, test_labels, *, data_iterator):
        if self.saver is None:
            self.define_model()
        if self.writer is None:
            self.writer = tf.summary.FileWriter('./board', tf.get_default_graph())

        print('Before session')
        with tf.Session(graph=tf.get_default_graph()) as session:
            self.saver.restore(session, self.save_path)
            ### 测试
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                result = session.run(
                    self.test_prediction,
                    feed_dict={self.tf_test_samples: samples}
                )
                # self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.print_confusion_matrix(np.add.reduce(confusionMatrices))
            ###

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        """
        计算预测的正确率与召回率
        @return: accuracy and confusionMatrix as a tuple
        """
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        # == is overloaded for numpy array
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm

    def visualize_filter_map(self, tensor, *, how_many, display_size, name):
        # print(tensor.get_shape)
        filter_map = tensor[-1]
        # print(filter_map.get_shape())
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        # print(filter_map.get_shape())
        filter_map = tf.reshape(filter_map, (how_many, display_size, display_size, 1))
        # print(how_many)
        self.test_summaries.append(tf.summary.image(name, tensor=filter_map, max_outputs=how_many))

    def print_confusion_matrix(self, confusionMatrix):
        print('Confusion    Matrix:')
        for i, line in enumerate(confusionMatrix):
            print(line, line[i] / np.sum(line))
        a = 0
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
            print(column[i] / np.sum(column), )
        print('\n', np.sum(confusionMatrix), a)
        
    def run_gradcam(self, img_path, label_idx, save_name='gradcam_result.png'):
        """
        计算并画出 Grad-CAM 热力图
        :param img_path: 图片路径
        :param label_idx: 目标类别的索引 (0-5), 比如饺子是3
        :param save_name: 保存文件名
        """
        print(f"Generating Grad-CAM for class {label_idx}...")
        
        # 1. 读取并预处理图片
        raw_img = cv2.imread(img_path)
        raw_img = cv2.resize(raw_img, (32, 32)) # 确保是 32x32
        # 归一化到 [-1, 1] 用于输入模型
        input_img = raw_img / 128.0 - 1.0 
        input_img = input_img.reshape(1, 32, 32, 3)

        # 2. 构建计算梯度的图 (Gradient Graph)
        # 目标是: 目标类别的 Logit 对 最后一层卷积特征图 的梯度
        target_logit = self.single_logits[0][label_idx]
        grads = tf.gradients(target_logit, self.last_conv_output)[0]
        
        # 3. 运行 Session 获取梯度和特征图
        # 注意: 必须在 restore 之后运行，所以通常在 test 模式下调用
        with tf.Session(graph=tf.get_default_graph()) as sess:
            self.saver.restore(sess, self.save_path)
            
            # 获取 梯度(grads_val) 和 特征图(conv_out_val)
            single_input_ph = tf.get_default_graph().get_tensor_by_name('test/single_input:0')
            
            grads_val, conv_out_val = sess.run(
                [grads, self.last_conv_output],
                feed_dict={single_input_ph: input_img}
            )
            
        # 4. 计算 Grad-CAM
        # conv_out_val shape: (1, 8, 8, 32) -> 去掉 batch 维度 -> (8, 8, 32)
        # grads_val shape: (1, 8, 8, 32)
        
        weights = np.mean(grads_val, axis=(0, 1, 2)) # Global Average Pooling on gradients -> (32,)
        cam = np.zeros(conv_out_val.shape[1:3], dtype=np.float32) # (8, 8)

        # 加权求和
        for i, w in enumerate(weights):
            cam += w * conv_out_val[0, :, :, i]
            
        # ReLU 激活
        cam = np.maximum(cam, 0)
        
        # 归一化到 0-1 之间
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        
        # 5. 可视化
        # 将热力图缩放回 32x32
        cam = cv2.resize(cam, (32, 32))
        
        # 转换为热力图颜色 (0-255)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 叠加: 原图 (0-255) + 热力图
        # raw_img 是 BGR (Opencv 默认)，转为 RGB 方便 matplotlib 显示
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # 热力图也转 RGB
        
        # 叠加权重: 0.5 原图 + 0.5 热力图
        superimposed_img = heatmap * 0.4 + rgb_img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        # 画图并保存
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 3, 1)
        plt.title('Original (32x32)')
        plt.imshow(rgb_img)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('Grad-CAM Heatmap')
        plt.imshow(heatmap)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Overlay')
        plt.imshow(superimposed_img)
        plt.axis('off')
        
        plt.savefig(save_name)
        print(f"Result saved to {save_name}")
        plt.close()
