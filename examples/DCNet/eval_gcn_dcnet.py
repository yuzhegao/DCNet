# coding: utf-8
import numpy as np
from PIL import Image
import scipy.io as sio
import os
import cv2
import time
import math

import os
os.environ['GLOG_minloglevel'] = '2'

# Make sure that caffe is on the python path:
caffe_root = '../../'
import sys

sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L

print("import caffe success")


def out_h(x):
    x_list = []
    x_list.append(x)
    x = math.floor((x - 3) / 2.0 + 1)
    x_list.append(x)
    x = math.floor((x - 1) / 2.0 + 1)
    x_list.append(x)
    x = math.floor((x - 1) / 2.0 + 1)
    x_list.append(x)
    x = math.floor((x - 1) / 2.0 + 1)
    x_list.append(x)
    # print(x_list)

    return x

def net_deploy(deploy_prototxt, checkpoint, input_hw, re_hw):
    from model.dcnet import dcnet

    n = caffe.NetSpec()
    n.data = L.Input(shape=[dict(dim=[1, 3, input_hw[0], input_hw[1]])])

    dcnet(n, is_train=False, re_hw1=re_hw)
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
# data_root = '../../data/BSDSownership/Augmentation/'
save_root = 'Output/dcnet/'
checkpoint = 'snapshot/dcnet_piod_iter_30000.caffemodel'





save_root = os.path.join(save_root, 'PIOD')
# save_root = os.path.join(save_root, 'BSDSownership')
if not os.path.exists(save_root):
    os.mkdir(save_root)

deploy_prototxt = 'dcnet_eval.prototxt'

# load net
caffe.set_mode_gpu()
# caffe.set_device(1)



with open(data_root + 'test.lst') as f:
    test_lst = f.readlines()

test_lst = [x.strip() for x in test_lst]

im_lst = []
gt_lst = []

for i in range(0, len(test_lst)):
    im = Image.open(test_lst[i])
    in_ = np.array(im, dtype=np.float32)
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
    re_h, re_w = out_h(im_.shape[1]), out_h(im_.shape[2])
    re_hw = [int(re_h), int(re_w)]
    print(im_.shape, re_hw)
    net = net_deploy(deploy_prototxt, checkpoint, input_hw=[im_.shape[1],im_.shape[2]], re_hw=re_hw)

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
    # plt.imshow(edgemap)
    # plt.show()
    cv2.imwrite(save_root + '/' + os.path.split(test_lst[idx])[1].split('.')[0] + '.png', edgemap * 255)
    sio.savemat(save_root + '/' + os.path.split(test_lst[idx])[1].split('.')[0] + '.mat', {'edge_ori': edge_ori})

tf = time.time()
print('total time: {}s'.format(tf-ts))
print('Detection took {:.3f}s per image'.format(time_total / len(test_lst)))
