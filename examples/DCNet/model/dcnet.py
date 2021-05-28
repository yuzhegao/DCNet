import sys

sys.path.append('python')

caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L, params as P

## False if TRAIN, True if TEST
bn_global_stats = False


def _conv_bn_scale(bottom, num_output, bias_term=False, **kwargs):
    '''Helper to build a conv -> BN -> relu block.
    '''
    global bn_global_stats
    conv = L.Convolution(bottom, num_output=num_output, bias_term=bias_term,
                         **kwargs)
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    return conv, bn, scale


def _conv_bn_scale_relu(bottom, num_output, bias_term=False, **kwargs):
    global bn_global_stats
    conv = L.Convolution(bottom, num_output=num_output, bias_term=bias_term,
                         **kwargs)
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    out_relu = L.ReLU(scale, in_place=True)

    return conv, bn, scale, out_relu


def _deconv_bn_scale_relu(bottom, nout, kernel_size, stride, pad, bias_term=False):
    ## just a bilinear upsample (lr_mult=0)
    global bn_global_stats

    conv = L.Deconvolution(bottom,
                           convolution_param=dict(num_output=nout, kernel_size=kernel_size, stride=stride, pad=pad,
                                                  bias_term=bias_term,
                                                  weight_filler={"type": "bilinear"}), param=[dict(lr_mult=0)])
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    out_relu = L.ReLU(scale, in_place=True)

    return conv, bn, scale, out_relu


def _conv_relu(bottom, nout, bias_term=False, **kwargs):
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term, **kwargs)
    out_relu = L.ReLU(conv, in_place=True)

    return out_relu, conv

def upsample_layer(bottom, uprate, in_dim):
    up_ = L.Deconvolution(bottom, convolution_param=dict(num_output=in_dim, kernel_size=uprate*2,
                                                                       group=in_dim,
                                                                       stride=uprate, pad=0, bias_term=False,
                                                                       weight_filler={"type": "bilinear"}),
                                                param=[dict(lr_mult=0)])
    return up_

encoder_dims = [64, 128, 128, 256]



def res50_convert(n, dims=encoder_dims):
    """
      Reduce channel of resnet output
    """
    _, _, _, n.conv1_convert = _conv_bn_scale_relu(n.conv1_relu, num_output=dims[0], bias_term=False, kernel_size=1,
                                                   stride=1, pad=0, weight_filler={"type": "msra"},
                                                   param=[dict(lr_mult=10)])
    _, _, _, n.res2c_convert = _conv_bn_scale_relu(n.res2c_relu, num_output=dims[1], bias_term=False, kernel_size=1,
                                                   stride=1, pad=0, weight_filler={"type": "msra"},
                                                   param=[dict(lr_mult=10)])
    _, _, _, n.res3d_convert = _conv_bn_scale_relu(n.res3d_relu, num_output=dims[2], bias_term=False, kernel_size=1,
                                                   stride=1, pad=0, weight_filler={"type": "msra"},
                                                   param=[dict(lr_mult=10)])
    _, _, _, n.res4f_convert = _conv_bn_scale_relu(n.res4f_relu, num_output=dims[3], bias_term=False, kernel_size=1,
                                                   stride=1, pad=0, weight_filler={"type": "msra"},
                                                   param=[dict(lr_mult=10)])


def res50_hl_convert(n, dims=[256]):
    """
      Reduce channel of resnet output
    """
    _, _, _, n.res4f_convert2 = _conv_bn_scale_relu(n.res4f_relu, num_output=dims[0], bias_term=False, kernel_size=1,
                                                   stride=1, pad=0, weight_filler={"type": "msra"},
                                                   param=[dict(lr_mult=10)])
    # _, _, _, n.res5c_convert2 = _conv_bn_scale_relu(n.res5c_relu, num_output=dims[0], bias_term=False, kernel_size=1,
    #                                                stride=1, pad=0, weight_filler={"type": "msra"},
    #                                                param=[dict(lr_mult=10)])


def gcn_module(n, bottom, in_channels=32, re_h=20, re_w=20):
    n.theta = L.Convolution(bottom, num_output=in_channels, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"},
                            bias_filler={"type": "constant", "value": 0.0},
                            param=[dict(lr_mult=10)])  # [N,C,H,W]

    n.phi = L.Convolution(bottom, num_output=in_channels, bias_term=True, kernel_size=1,
                          stride=1, pad=0, weight_filler={"type": "msra"},
                          bias_filler={"type": "constant", "value": 0.0},
                          param=[dict(lr_mult=10)])  # [N,C,H,W]

    n.G = L.Convolution(bottom, num_output=in_channels, bias_term=True, kernel_size=1,
                        stride=1, pad=0, weight_filler={"type": "msra"},
                        bias_filler={"type": "constant", "value": 0.0},
                        param=[dict(lr_mult=10)])  # [N,C,H,W]

    n.theta_reshape = L.Reshape(n.theta, reshape_param={'shape': {'dim': [0, 0, -1]}})  # [N,C,H*W]
    n.phi_reshape = L.Reshape(n.phi, reshape_param={'shape': {'dim': [0, 0, -1]}})  # [N,C,H*W]
    n.G_reshape = L.Reshape(n.G, reshape_param={'shape': {'dim': [0, 0, -1]}})  # [N,C,H*W]

    n.theta_transpose = L.TensorTranspose(n.theta_reshape, tensor_transpose_param={'order': [0, 2, 1]})  # [N,H*W,C]
    n.G_transpose = L.TensorTranspose(n.G_reshape, tensor_transpose_param={'order': [0, 2, 1]})  # [N,H*W,C]
    n.ajacent = L.MatrixMultiplication(n.theta_transpose, n.phi_reshape)  # [N,H*W,H*W]

    n.ajacent_softmax = L.Softmax(n.ajacent, softmax_param={'axis': 2})  # [N,H*W,H*W]
    n.G_round = L.MatrixMultiplication(n.ajacent_softmax, n.G_transpose)  # [N,H*W,C]
    n.G_round_transpose = L.TensorTranspose(n.G_round, tensor_transpose_param={'order': [0, 2, 1]})  # [N,C,H*W]

    n.G_reshape2 = L.Reshape(n.G_round_transpose, reshape_param={'shape': {'dim': [0, 0, re_h, re_w]}})  # [N,C,H,W]

    n.W_conv, n.W_bn, n.W_scale = _conv_bn_scale(n.G_reshape2, num_output=in_channels, kernel_size=1,
                                                 stride=1, pad=0, weight_filler={"type": "msra"},
                                                 param=[dict(lr_mult=10)])



def BCP_module(n, bottom1, bottom2, in_channels, out_channels, re_h1, re_w1, re_h2, re_w2):
    """
     bottom1: the feature of complete boundary
     bottom2: the 2nd-stage of backbone
    """
    n.N2_conv = L.Convolution(bottom2, num_output=out_channels, bias_term=True, kernel_size=1,
                              stride=1, pad=0, weight_filler={"type": "msra"},
                              bias_filler={"type": "constant", "value": 0.0},
                              param=[dict(lr_mult=10)])  # [N,C2,H,W]

    n.N2_mask = L.Convolution(bottom2, num_output=1, bias_term=True, kernel_size=1,
                              stride=1, pad=0, weight_filler={"type": "msra"},
                              bias_filler={"type": "constant", "value": 0.0},
                              param=[dict(lr_mult=10)])  # [N,1,H,W]
    n.N2_mask_sigmoid = L.Sigmoid(n.N2_mask)
    n.N2_mask_tile = L.Tile(n.N2_mask_sigmoid, tile_param={"axis": 1, "tiles": in_channels})
    n.N2_mask_reverse = L.Power(n.N2_mask_tile, power_param={'power': -1,
                                                             'scale': 1,
                                                             'shift': 1})  # reverse



    n.theta = L.Convolution(bottom1, num_output=in_channels, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"},
                            bias_filler={"type": "constant", "value": 0.0},
                            param=[dict(lr_mult=10)])  # [N,C1,H,W]
    n.theta1 = L.Eltwise(n.theta, n.N2_mask_tile, eltwise_param={'operation': 0})
    n.theta2 = L.Eltwise(n.theta, n.N2_mask_reverse, eltwise_param={'operation': 0})

    n.phi = L.Convolution(bottom1, num_output=in_channels, bias_term=True, kernel_size=1,
                          stride=1, pad=0, weight_filler={"type": "msra"},
                          bias_filler={"type": "constant", "value": 0.0},
                          param=[dict(lr_mult=10)])  # [N,C1,H,W]
    n.phi1 = L.Eltwise(n.phi, n.N2_mask_tile, eltwise_param={'operation': 0})
    n.phi2 = L.Eltwise(n.phi, n.N2_mask_reverse, eltwise_param={'operation': 0})

    n.G = L.Convolution(bottom1, num_output=in_channels, bias_term=True, kernel_size=1,
                        stride=1, pad=0, weight_filler={"type": "msra"},
                        bias_filler={"type": "constant", "value": 0.0},
                        param=[dict(lr_mult=10)])  # [N,C1,H,W]

    ## mask ajacent
    n.theta_reshape = L.Reshape(n.theta1, reshape_param={'shape': {'dim': [0, 0, -1]}})  # [N,C,H*W]
    n.phi_reshape = L.Reshape(n.phi1, reshape_param={'shape': {'dim': [0, 0, -1]}})  # [N,C,H*W]
    n.theta_transpose = L.TensorTranspose(n.theta_reshape, tensor_transpose_param={'order': [0, 2, 1]})  # [N,H*W,C]

    n.ajacent = L.MatrixMultiplication(n.theta_transpose, n.phi_reshape)  # [N,H*W,H*W]
    n.ajacent_softmax = L.Softmax(n.ajacent, softmax_param={'axis': 2})  # [N,H*W,H*W]


    ## reverse mask ajacent
    n.theta_reshape2 = L.Reshape(n.theta2, reshape_param={'shape': {'dim': [0, 0, -1]}})  # [N,C,H*W]
    n.phi_reshape2 = L.Reshape(n.phi2, reshape_param={'shape': {'dim': [0, 0, -1]}})  # [N,C,H*W]
    n.theta_transpose2 = L.TensorTranspose(n.theta_reshape2, tensor_transpose_param={'order': [0, 2, 1]})  # [N,H*W,C]

    n.ajacent2 = L.MatrixMultiplication(n.theta_transpose2, n.phi_reshape2)  # [N,H*W,H*W]
    n.ajacent_softmax2 = L.Softmax(n.ajacent2, softmax_param={'axis': 2})  # [N,H*W,H*W]


    ## reshape the N2 tensor
    n.N2_reshape = L.Reshape(n.G, reshape_param={'shape': {'dim': [0, 0, -1]}})  # [N,C2,H2*W2]
    n.N2_transpose = L.TensorTranspose(n.N2_reshape, tensor_transpose_param={'order': [0, 2, 1]})  # [N,H*W,C2]

    ## mask output tensor
    n.N2_weightadd = L.MatrixMultiplication(n.ajacent_softmax, n.N2_transpose)  # [N,H*W,C2]
    n.N2_weightadd_transpose = L.TensorTranspose(n.N2_weightadd,
                                                 tensor_transpose_param={'order': [0, 2, 1]})  # [N,C2,H*W]
    n.N2_weightadd_reshape = L.Reshape(n.N2_weightadd_transpose,
                                       reshape_param={'shape': {'dim': [0, 0, re_h1, re_w1]}})  # [N,C,H1,W1]

    ## reverse mask output tensor
    n.N2_weightadd2 = L.MatrixMultiplication(n.ajacent_softmax2, n.N2_transpose)  # [N,H*W,C2]
    n.N2_weightadd_transpose2 = L.TensorTranspose(n.N2_weightadd2,
                                                 tensor_transpose_param={'order': [0, 2, 1]})  # [N,C2,H*W]
    n.N2_weightadd_reshape2 = L.Reshape(n.N2_weightadd_transpose2,
                                       reshape_param={'shape': {'dim': [0, 0, re_h1, re_w1]}})  # [N,C,H1,W1]

    n.BCP_out = L.Eltwise(n.N2_conv, n.N2_weightadd_reshape, n.N2_weightadd_reshape2)


def output_branch(n, bottom, task):
    """ For edge and ori output (task: edge or ori)
        edge and ori has same branch [8,8,8,8,4,1]
    """
    conv = 'unet1a_plain1a_conv_{}'.format(task)
    bn = 'unet1a_plain1a_bn_{}'.format(task)
    scale = 'unet1a_plain1a_scale_{}'.format(task)
    relu1 = 'unet1a_plain1a_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu1] = _conv_bn_scale_relu(bottom, num_output=8, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0},
                                                             param=[dict(lr_mult=10)])



    conv = 'unet1b_plain1b_conv_{}'.format(task)
    bn = 'unet1b_plain1b_bn_{}'.format(task)
    scale = 'unet1b_plain1b_scale_{}'.format(task)
    relu5 = 'unet1b_plain1b_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu5] = _conv_bn_scale_relu(n[relu1], num_output=4, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0},
                                                             param=[dict(lr_mult=10)])

    conv = 'unet1b_{}'.format(task)
    n[conv] = L.Convolution(n[relu5], num_output=1, bias_term=True, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"},
                            bias_filler={"type": "constant", "value": 0.0},
                            param=[dict(lr_mult=10)])


def _resnet_path(name, n, bottom, nout1, nout2, nout3, branch1=False, initial_stride=1):
    '''Basic ResNet block.
    '''
    if branch1:  ## if is branch1 in a stage, should change dim (like downsample in pytorch)
        res_b1 = 'res{}_b1'.format(name)
        bn_b1 = 'bn{}_b1'.format(name)
        scale_b1 = 'scale{}_b1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, nout3, kernel_size=1, stride=initial_stride, pad=0,
            weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])
        ## don't forget param: weight_filler={"type": "xavier"},bias_filler={"type": "constant"},
        ## param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)  (if need)
    else:
        initial_stride = 1

    res = 'res{}_b2a'.format(name)
    bn = 'bn{}_b2a'.format(name)
    scale = 'scale{}_b2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout1, kernel_size=1, stride=initial_stride, pad=0,
        weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])
    relu2a = 'res{}_b2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_b2b'.format(name)
    bn = 'bn{}_b2b'.format(name)
    scale = 'scale{}_b2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout2, kernel_size=3, stride=1, pad=1,
        weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])
    relu2b = 'res{}_b2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_b2c'.format(name)
    bn = 'bn{}_b2c'.format(name)
    scale = 'scale{}_b2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], nout3, kernel_size=1, stride=1, pad=0,
        weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)

    return n[relu]


def _resnet_block(name, n, bottom, nout, branch1=False, initial_stride=2):
    '''Basic ResNet block.
    '''
    if branch1:  ## if is branch1 in a stage, should change dim (like downsample in pytorch)
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, 4 * nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
        ## don't forget param: weight_filler={"type": "xavier"},bias_filler={"type": "constant"},
        ## param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)  (if need)
    else:
        initial_stride = 1

    res = 'res{}_branch2a'.format(name)
    bn = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
    relu2a = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=1, weight_filler={"type": "msra"})
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], 4 * nout, kernel_size=1, stride=1, pad=0, weight_filler={"type": "msra"})
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)


def _resnet_dilation_block(name, n, bottom, nout, branch1=False, initial_stride=2, dil_rate=2):
    '''Basic ResNet block.
    '''
    if branch1:
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, 4 * nout, kernel_size=1, stride=initial_stride,
            pad=0, weight_filler={"type": "msra"})
    else:
        initial_stride = 1

    res = 'res{}_branch2a'.format(name)
    bn = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
    relu2a = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    # dilation
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=dil_rate, weight_filler={"type": "msra"}, dilation=dil_rate)
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], 4 * nout, kernel_size=1, stride=1, pad=0, weight_filler={"type": "msra"})
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)


def resnet50_branch(n, bottom):
    '''ResNet 50 layers.
    '''
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(
        bottom, 64, bias_term=True, kernel_size=7, pad=3, stride=2, weight_filler={"type": "msra"})
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(
        n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # stage 2
    _resnet_block('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block('2b', n, n.res2a_relu, 64)
    _resnet_block('2c', n, n.res2b_relu, 64)  # res2c_relu

    # stage 3
    _resnet_block('3a', n, n.res2c_relu, 128, branch1=True)
    _resnet_block('3b', n, n.res3a_relu, 128)
    _resnet_block('3c', n, n.res3b_relu, 128)
    _resnet_block('3d', n, n.res3c_relu, 128)  # res3d_relu

    # stage 4
    _resnet_block('4a', n, n.res3d_relu, 256, branch1=True)
    _resnet_block('4b', n, n.res4a_relu, 256)
    _resnet_block('4c', n, n.res4b_relu, 256)
    _resnet_block('4d', n, n.res4c_relu, 256)
    _resnet_block('4e', n, n.res4d_relu, 256)
    _resnet_block('4f', n, n.res4e_relu, 256)  # res4f_relu

    # stage 5
    _resnet_dilation_block('5a', n, n.res4f_relu, 512, branch1=True, initial_stride=1, dil_rate=2)
    _resnet_dilation_block('5b', n, n.res5a_relu, 512, dil_rate=4)
    _resnet_dilation_block('5c', n, n.res5b_relu, 512, dil_rate=8)  # res5c_relu


def BRU_block(name1, n, bottom1, bottom2, in_dim1, out_dim, side_up, up_2=True, next=True):
    ## name1 :  4f 3d 2c 1a
    if up_2:
        n[name1 + '_rescale'] = L.Deconvolution(bottom1,
                                                convolution_param=dict(num_output=in_dim1, kernel_size=4,
                                                                       group=in_dim1,
                                                                       stride=2, pad=0, bias_term=False,
                                                                       weight_filler={"type": "bilinear"}),
                                                param=[dict(lr_mult=0)])
        bottom1 = n[name1 + '_rescale']

    crop_bottoms = [bottom1, bottom2]
    n[name1 + '_rescale1'] = L.Crop(*crop_bottoms, crop_param={'axis': 2})

    # n[name1 + '_refine_relu'] = _resnet_path(name1, n, n[name1 + '_rescale1'],
    #                                          nout1=in_dim1, nout2=in_dim1, nout3=in_dim1, branch1=True)
    n[name1 + '_refine'], _, _, n[name1 + '_refine_relu'] = _conv_bn_scale_relu(n[name1 + '_rescale1'],
                                                                                num_output=in_dim1, bias_term=False,
                                                                                kernel_size=3,
                                                                                stride=1, pad=1,
                                                                                weight_filler={"type": "msra"},
                                                                                param=[dict(lr_mult=10)])

    n[name1 + '_mul'] = L.Eltwise(n[name1 + '_refine_relu'], bottom2, eltwise_param={'operation': 0})  ## multiply
    n[name1 + '_refine2'], _, _, n[name1 + '_refine2_relu'] = _conv_bn_scale_relu(n[name1 + '_mul'],
                                                                                  num_output=in_dim1, bias_term=False,
                                                                                  kernel_size=3,
                                                                                  stride=1, pad=1,
                                                                                  weight_filler={"type": "msra"},
                                                                                  param=[dict(lr_mult=10)])

    n[name1 + '_mul_add'] = L.Eltwise(n[name1 + '_rescale1'], n[name1 + '_refine2_relu'])

    n[name1 + '_fuse'], _, _, n[name1 + '_fuse_relu'] = _conv_bn_scale_relu(n[name1 + '_mul_add'],
                                                                            num_output=out_dim, bias_term=False,
                                                                            kernel_size=3,
                                                                            stride=1, pad=1,
                                                                            weight_filler={"type": "msra"},
                                                                            param=[dict(lr_mult=10)])


def edge_path(n, bottom, re_hw1, re_hw2):
    ## FPN architecture
    BRU_block('res4f', n, bottom, n.res4f_convert, in_dim1=encoder_dims[3], out_dim=encoder_dims[2],
              side_up=16, up_2=False)

    BCP_module(n, bottom1=n.encoder_hl_relu, bottom2=n.res4f_fuse_relu,
               in_channels=128, out_channels=encoder_dims[1],
               re_h1=re_hw1[0], re_w1=re_hw1[1], re_h2=re_hw2[0], re_w2=re_hw2[1])

    BRU_block('res3d', n, n.BCP_out, n.res3d_convert, in_dim1=encoder_dims[2], out_dim=encoder_dims[1],
              side_up=8)

    BRU_block('res2c', n, n.res3d_fuse, n.res2c_convert, in_dim1=encoder_dims[1], out_dim=encoder_dims[0],
              side_up=4)
    BRU_block('res1a', n, n.res2c_fuse_relu, n.conv1_convert, in_dim1=encoder_dims[0], out_dim=64,
              side_up=2)
    n.res1a_rescale1 = L.Deconvolution(n.res1a_fuse_relu,
                                       convolution_param=dict(num_output=64, kernel_size=4,
                                                              group=64,
                                                              stride=2, pad=0, bias_term=False,
                                                              weight_filler={"type": "bilinear"}),
                                       param=[dict(lr_mult=0)])
    crop_bottoms = [n.res1a_rescale1, n.data]
    n.res1a_rescale2 = L.Crop(*crop_bottoms, crop_param={'axis': 2})


def _msl_block2(n, bottom, dil_rates=[2, 3, 6], out_dim=256):
    n.aspp_1, _, _, n.relu_aspp_1 = _conv_bn_scale_relu(bottom, num_output=out_dim, bias_term=False,
                                                        kernel_size=1, stride=1, pad=0, weight_filler={"type": "msra"},
                                                        param=[dict(lr_mult=10)])
    n.aspp_2, _, _, n.relu_aspp_2, = _conv_bn_scale_relu(bottom, num_output=out_dim, kernel_size=3,
                                                         dilation=dil_rates[0],
                                                         pad=dil_rates[0], stride=1, weight_filler={"type": "msra"},
                                                         param=[dict(lr_mult=10)])
    n.aspp_3, _, _, n.relu_aspp_3, = _conv_bn_scale_relu(bottom, num_output=out_dim, kernel_size=3,
                                                         dilation=dil_rates[1],
                                                         pad=dil_rates[1], stride=1, weight_filler={"type": "msra"},
                                                         param=[dict(lr_mult=10)])
    n.aspp_4, _, _, n.relu_aspp_4, = _conv_bn_scale_relu(bottom, num_output=out_dim, kernel_size=3,
                                                         dilation=dil_rates[2],
                                                         pad=dil_rates[2], stride=1, weight_filler={"type": "msra"},
                                                         param=[dict(lr_mult=10)])

    concat_layers = [n.relu_aspp_1, n.relu_aspp_2, n.relu_aspp_3, n.relu_aspp_4]
    n.aspp_concat = caffe.layers.Concat(*concat_layers, concat_param=dict(concat_dim=1))

    n.aspp_refine, _, _, n.aspp_refine_relu = _conv_bn_scale_relu(n.aspp_concat, num_output=out_dim, kernel_size=1,
                                                                  stride=1, weight_filler={"type": "msra"},
                                                                  param=[dict(lr_mult=10)])

    n.aspp_refine2, _, _, n.aspp_refine2_relu = _conv_bn_scale_relu(n.aspp_refine_relu, num_output=16,
                                                                    kernel_size=1,
                                                                    stride=1, weight_filler={"type": "msra"},
                                                                    param=[dict(lr_mult=10)])


def ori_path(n, bottom):
    ## from high-level multi-scale feature
    _msl_block2(n, bottom)
    # n.aspp_up1, n.aspp_up1_bn, n.aspp_up1_scale, n.aspp_up1_relu = \
    #     _deconv_bn_scale_relu(n.aspp_concat, nout=32, kernel_size=8, stride=4, pad=0)
    #
    # crop_bottoms = [n.aspp_up1_relu, n.res2c_relu]
    # n.aspp_crop1 = L.Crop(*crop_bottoms, crop_param={'axis': 2})  ## crop 1/4
    #
    # n.aspp_up2, n.aspp_up2_bn, n.aspp_up2_scale, n.aspp_up2_relu = \
    #     _deconv_bn_scale_relu(n.aspp_crop1, nout=32, kernel_size=8, stride=4, pad=0)
    #
    n.aspp_rescale = L.Deconvolution(n.aspp_refine2_relu,
                                     convolution_param=dict(num_output=16, kernel_size=32,
                                                            group=16,
                                                            stride=16, pad=0, bias_term=False,
                                                            weight_filler={"type": "bilinear"}),
                                     param=[dict(lr_mult=0)])
    crop_bottoms = [n.aspp_rescale, n.data]
    n.aspp_crop2 = L.Crop(*crop_bottoms, crop_param={'axis': 2})  ## crop 1

    ## from boundary-like cue
    n.ori_branch_feat_1_conv, _, _, n.ori_branch_feat_1_relu = \
        _conv_bn_scale_relu(n.plain7b_relu, num_output=16, bias_term=False, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])

    n.ori_branch_feat_rescale1 = L.Deconvolution(n.ori_branch_feat_1_relu,
                                                 convolution_param=dict(num_output=16, kernel_size=8,
                                                                        group=16,
                                                                        stride=4, pad=0, bias_term=False,
                                                                        weight_filler={"type": "bilinear"}),
                                                 param=[dict(lr_mult=0)])

    n.ori_branch_feat_2_conv, _, _, n.ori_branch_feat_2_relu = \
        _conv_bn_scale_relu(n.ori_branch_feat_rescale1, num_output=8, bias_term=False, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])
    n.ori_branch_feat_rescale2 = L.Deconvolution(n.ori_branch_feat_2_relu,
                                                 convolution_param=dict(num_output=8, kernel_size=8,
                                                                        group=8,
                                                                        stride=4, pad=0, bias_term=False,
                                                                        weight_filler={"type": "bilinear"}),
                                                 param=[dict(lr_mult=0)])

    ori_branch_feat_crop_bottoms = [n.ori_branch_feat_rescale2, n.data]
    n.ori_branch_feat_rescale3 = L.Crop(*ori_branch_feat_crop_bottoms, crop_param={'axis': 2})

    n.ori_concat = L.Concat(n.ori_branch_feat_rescale3, n.aspp_crop2, n.unet1a_plain1a_relu_edge,
                            concat_param=dict(concat_dim=1), propagate_down=[1, 1, 0])


def dcnet(n, is_train=True, re_hw1=[20, 20], re_hw2=[40, 40]):
    global bn_global_stats
    bn_global_stats = False if is_train else True

    resnet50_branch(n, n.data)  ## resnet50 backbone
    res50_convert(n)
    res50_hl_convert(n, dims=[256])


    ## plain 6
    n.plain6a_conv, n.plain6a_bn, n.plain6a_scale, n.plain6a_relu = \
        _conv_bn_scale_relu(n.res5c_relu, num_output=256, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])

    n.plain6b_conv, n.plain6b_bn, n.plain6b_scale, n.plain6b_relu = \
        _conv_bn_scale_relu(n.plain6a_relu, num_output=256, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])
    ## plain 7
    n.plain7a_conv, n.plain7a_bn, n.plain7a_scale, n.plain7a_relu = \
        _conv_bn_scale_relu(n.res5c_relu, num_output=256, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])

    n.plain7b_conv, n.plain7b_bn, n.plain7b_scale, n.plain7b_relu = \
        _conv_bn_scale_relu(n.plain7a_relu, num_output=256, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"}, param=[dict(lr_mult=10)])

    n.encoder_hl_concat = L.Concat(n.plain6b_relu, n.res4f_convert2)
    n.encoder_hl, _, _, n.encoder_hl_relu = _conv_bn_scale_relu(n.encoder_hl_concat, num_output=256, bias_term=False,
                                                                kernel_size=1,
                                                                stride=1, pad=0, weight_filler={"type": "msra"},
                                                                param=[dict(lr_mult=10)])

    ## edge path
    edge_path(n, n.plain6b_relu, re_hw1, re_hw2)
    output_branch(n, n.res1a_rescale2, task='edge')

    ## ori path
    ori_path(n, n.plain7b_relu)
    output_branch(n, n.ori_concat, task='ori')


if __name__ == '__main__':
    n = caffe.NetSpec()
    # n.data = L.DummyData(shape=[dict(dim=[1, 3, 224, 224])])
    n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])

    dcnet(n, is_train=True, re_hw1=[56, 56], re_hw2=[28,28])
    with open('dcnet_example.prototxt', 'w') as f:
        f.write(str(n.to_proto()))

