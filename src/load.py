from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np


def reformat(samples, labels):
    # 改变原始数据的形状
    #  0       1       2      3          3       0       1      2
    # (图片高，图片宽，通道数，图片数) -> (图片数，图片高，图片宽，通道数)
    new = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)

    # labels 变成 one-hot encoding, [2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # digit 0 , represented as 10
    # labels 变成 one-hot encoding, [10] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = np.array([x[0] for x in labels])  # slow code, whatever
    one_hot_labels = []
    for num in labels:
        # 改成 6 
        one_hot = [0.0] * 6  
        # 直接赋值，标签已经是 0,1,2,3,4,5 
        one_hot[int(num)] = 1.0 
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels


def normalize(samples):
    """
    保留彩色通道 (RGB)
    将图片从 0 ~ 255 线性映射到 -1.0 ~ +1.0
    """
    # 直接除以 128.0 减 1.0，不合并通道
    return samples / 128.0 - 1.0

def distribution(labels, name):
    # 查看一下每个label的分布，再画个统计图
    # keys:
    # 0
    # 1
    # 2
    # ...
    # 9
    count = { }
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    x = []
    y = []
    for k, v in count.items():
        # print(k, v)
        x.append(k)
        y.append(v)

    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()


def inspect(dataset, labels, i):
    # 显示图片看看
    if dataset.shape[3] == 1:
        shape = dataset.shape
        dataset = dataset.reshape(shape[0], shape[1], shape[2])
    print(labels[i])
    plt.imshow(dataset[i])
    plt.show()


train = load('train.mat')
test = load('test.mat')


# print('Train Samples Shape:', train['X'].shape)
# print('Train  Labels Shape:', train['y'].shape)

# print('Train Samples Shape:', test['X'].shape)
# print('Train  Labels Shape:', test['y'].shape)


train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']


n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)

num_labels = 6
image_size = 32
num_channels = 3

if __name__ == '__main__':
    # 探索数据
    pass
    inspect(_train_samples, _train_labels, 1234)
# _train_samples = normalize(_train_samples)
# inspect(_train_samples, _train_labels, 1234)
# distribution(train_labels, 'Train Labels')
# distribution(test_labels, 'Test Labels')
