# coding: utf-8
import numpy as np
from PIL import Image
import scipy.io as sio
import os
import cv2
import time

os.environ['GLOG_minloglevel'] = '2'

# Make sure that caffe is on the python path:
caffe_root = '../../'
import sys
import math

sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L
print("import caffe success")


def out_h(x):
    x_list = []
    x_list.append(x)
    x = math.floor((x - 3) / 2.0 + 1) ## 1/2
    x_list.append(x)
    x = math.floor((x - 1) / 2.0 + 1) ## 1/4
    x_list.append(x)
    x = math.floor((x - 1) / 2.0 + 1) ## 1/8
    x_list.append(x)
    x = math.floor((x - 1) / 2.0 + 1) ## 1/16
    x_list.append(x)
    # print(x_list)

    return x_list

def net_deploy(deploy_prototxt, checkpoint, input_hw, re_hw1, re_hw2):
    from model.ofnet_92_29 import ofnet

    n = caffe.NetSpec()
    n.data = L.Input(shape=[dict(dim=[1, 3, input_hw[0], input_hw[1]])])

    # ofnet(n, is_train=False, re_hw=re_hw1, re_hw1=re_hw1, re_hw2=re_hw2)
    # ofnet(n, is_train=False, re_hw=re_hw1)
    ofnet(n, is_train=False, re_hw1=re_hw1, re_hw2=re_hw2)
    n.sigmoid_edge = L.Sigmoid(n.unet1b_edge)
    # n.sigmoid_edge = L.Sigmoid(n.edge_p2)

    with open(deploy_prototxt, 'w') as f:
        f.write(str(n.to_proto()))  ## write network

    net = caffe.Net(deploy_prototxt,
                    checkpoint,
                    caffe.TEST)
    return net


## should change the [model path] + [save_path] + [import module]

data_root = '../../data/PIOD/Augmentation/'
checkpoint = 'snapshot/ofnet_92_29_iter_30000.caffemodel'
save_path = '/home/gyz/data/kitti_train/occ_res/'

deploy_prototxt = 'ofnet_eval.prototxt'

# load net
caffe.set_mode_gpu()
caffe.set_device(3)



import glob
test_lst = glob.glob('/home/gyz/data/kitti_train/image_2/*.png')
test_lst = test_lst[:300]
print(test_lst)
# test_lst = ['000033.png', '000080.png', '000092.png', '000159.png']

im_lst = []
gt_lst = []

for i in range(0, len(test_lst)):
    im = Image.open(test_lst[i])
    in_ = np.array(im, dtype=np.float32)[:,:,:3]
    print(in_.shape)
    is_crop = False
    if is_crop:
        in_ = in_[:, 300:900, :]
    cv2.imwrite(save_path + os.path.split(test_lst[i])[1].split('.')[0] + '_crop.png', in_)
    in_ = in_[:, :, ::-1]
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    im_lst.append(in_)

time_total = 0

ts = time.time()
for idx in range(0, len(test_lst)):
    print(idx)
    im_ = im_lst[idx]
    im_ = im_.transpose((2, 0, 1))
    # print(im_.shape)
    h_list, w_list = out_h(im_.shape[1]), out_h(im_.shape[2])
    re_hw1 = [int(h_list[4]), int(w_list[4])]
    # re_hw1 = [int(h_list[2]), int(w_list[2])]
    re_hw2 = [int(h_list[3]), int(w_list[3])]
    print(im_.shape, re_hw1, re_hw2)
    # print(im_.shape, re_hw)
    net = net_deploy(deploy_prototxt, checkpoint, input_hw=[im_.shape[1], im_.shape[2]], re_hw1=re_hw1, re_hw2=re_hw2)

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im_.shape)
    net.blobs['data'].data[...] = im_
    # run net and take argmax for prediction
    start_time = time.time()
    net.forward()
    diff_time = time.time() - start_time
    time_total += diff_time

    edgemap = net.blobs['sigmoid_edge'].data[0][0, :, :]
    orimap = net.blobs['unet1b_ori'].data[0][0, :, :]
    # print edgemap


    edge_ori = {}
    edge_ori['edge'] = edgemap
    edge_ori['ori'] = orimap
    cv2.imwrite(save_path + os.path.split(test_lst[idx])[1].split('.')[0] + '_.png', edgemap * 255)
    sio.savemat(save_path + os.path.split(test_lst[idx])[1].split('.')[0] + '_.mat', {'edge_ori': edge_ori})

# diff_time = time.time() - start_time
# print('Detection took {:.3f}s per image'.format(diff_time / len(test_lst)))
