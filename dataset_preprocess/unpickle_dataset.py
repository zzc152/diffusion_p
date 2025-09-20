import os
import numpy as np
import pickle
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch_torch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_' + str(idx))
    d = unpickle(data_file)
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x.astype(np.float32) / 255.0
    mean_image = mean_image.astype(np.float32) / 255.0

    # 标签从1开始，整理成从0开始
    y = [i-1 for i in y]

    data_size = x.shape[0]
    x -= mean_image  # 逐像素减去均值
    img_size2 = img_size * img_size

    # 将每个样本从一维数组变成 H x W x C
    x = np.dstack(
        (x[:, :img_size2],
         x[:, img_size2:2*img_size2],
         x[:, 2*img_size2:])
    )
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    # 转置为 (N, C, H, W)
    x = x.transpose(0, 3, 1, 2)

    # 转为torch tensor
    X = torch.from_numpy(x)
    Y = torch.tensor(y, dtype=torch.int64)

    # 镜像扩增：水平翻转
    X_flip = torch.flip(X, dims=[3])  # 逆序宽度维度
    Y_flip = Y  # 标签不变

    # 拼接
    X_combined = torch.cat([X, X_flip], dim=0)
    Y_combined = torch.cat([Y, Y_flip], dim=0)

    # 转为float，包括归一化
    X_combined = X_combined.float()

    return {
        'X_train': X_combined,
        'Y_train': Y_combined,
        'mean': torch.tensor(mean_image )
    }


def display_picture(n_row, n_column, picture_tensor):
    picture_tensor = picture_tensor[0:n_row*n_column, :, :, :]
    grid_img = vutils.make_grid(picture_tensor, nrow=n_row, padding=2)

    # 转换为 numpy 数组，方便用 matplotlib 显示
    np_img = grid_img.permute(1, 2, 0).cpu().numpy()

    plt.imshow(np.clip(np_img, 0, 1))
    plt.axis('off')
    plt.show()


dict = load_databatch_torch("../train", 1, img_size=32)
x = dict['X_train']
y = dict['Y_train']
print(x.shape)
print(y.shape)
mean = dict['mean'].reshape(32, 32, 3)
mean = mean.transpose(0, 2).transpose(1, 2)
print(mean.shape)
display_picture(4, 4, x+ mean)
display_picture(4, 4, x)
