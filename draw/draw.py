import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import argparse


parser = argparse.ArgumentParser(description='draw pic')
parser.add_argument('--feature',type=str)
parser.add_argument('--label',type=str)
parser.add_argument('--out-folder',type=str)
parser.add_argument('--sname',type=str)

args = parser.parse_args()

# class
ignore_rows_rate = 2
features = np.load(args.feature)  # 将�?��?�的�?径改为embeding的路�?
# print(features)
# print(features.shape)
features = features[0:64]
features = np.squeeze(features)
labels = np.load(args.label)  # 将�?��?�的�?径改为label的路径，label�?0�?1�?2�?...
# print(labels)
# print(labels.shape)
labels = labels[0:64]
pic_feature = features.copy()
idx_pf = 0
de = 4

if not os.path.exists(args.out_folder):
    os.mkdir(args.out_folder)

pic_feature_mini = np.ones((pic_feature.shape[0], pic_feature.shape[1] // de))
for i in range(pic_feature.shape[1] // de):
    pic_feature_mini[:, i] = np.sum(pic_feature[:, i * de:i * de + de], axis=1)

pic_feature = pic_feature_mini
features = pic_feature
for i in range(5):  # 将各个label对应的embeding归组
    labels = np.squeeze(labels)
    idxs = np.argwhere(labels == i)
    idxs = np.squeeze(idxs)
    pic_feature[idx_pf:(idx_pf + idxs.shape[0])] = features[idxs].copy()
    show_pic = pic_feature[idx_pf:(idx_pf + 50)]
    fig, (ax0) = plt.subplots(1)
    c = ax0.pcolor(show_pic)
    ax0.set_title('default: no edges')
    fig.tight_layout()
    plt.savefig(os.path.join(args.out_folder,'rep_class_{}.jpeg'.format(i)))
    # plt.show()
    idx_pf = idx_pf + idxs.shape[0]
cv.imwrite(os.path.join(args.out_folder,'GrayImage2.png'), pic_feature * 100)  # 输出图片
cv.waitKey()
fig, (ax0, ax1) = plt.subplots(1, 2)
c = ax0.pcolor(features)
ax0.set_title('default: no edges')
c = ax1.pcolor(features, edgecolors='k', linewidths=4)
ax1.set_title('thick edges')
fig.tight_layout()
plt.show()

# rgb
ignore_rows_rate=2
features = np.load(args.feature)
features = features[0:64]
features = np.squeeze(features)
labels = np.load(args.label)
labels = labels[0:64]
pic_feature = features.copy()
idx_pf = 0
for i in range(5):
    idxs = np.argwhere(labels == i)
    idxs = np.squeeze(idxs)
    s = slice(0, idxs.shape[0] - 1, ignore_rows_rate)
    idxs = idxs[s]

    pic_feature[idx_pf:(idx_pf + idxs.shape[0])] = features[idxs].copy()

    idx_pf = idx_pf + idxs.shape[0]

cv.imwrite(os.path.join(args.out_folder,'GrayImage.png'), pic_feature * 10000)
cv.imwrite(os.path.join(args.out_folder,'GrayImagef.png'), features * 10000)
cv.waitKey()

delete_row_n = (pic_feature.shape[1] % 3)
pic_feature_RGB = pic_feature
for j in range(delete_row_n):
    pic_feature_RGB = np.delete(pic_feature_RGB, 1, axis=1)
a_pic = pic_feature_RGB.shape[1]
pic_feature_RGB = pic_feature_RGB.reshape(pic_feature_RGB.shape[0], a_pic // 3, 3)
cv.imwrite(os.path.join(args.out_folder,'RGBImage.png'), pic_feature_RGB * 100000)
cv.waitKey()