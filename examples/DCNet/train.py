import sys
sys.path.append('python')

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
print('import caffe success')

import caffe
from caffe import layers as L, params as P

from model.dcnet import dcnet

def write_solver(base_lr=0.00003, iters=30000, snapshot='snapshot/dcnet'):
    """
    Generate solver.prototxt.
    base_lr: learning rate;
    iters: max iterations;
    snapshot: the prefix of saved models.
    """
    sovler_string = caffe.proto.caffe_pb2.SolverParameter() ## define solver

    sovler_string.net = 'dcnet.prototxt'
    sovler_string.test_iter.append(0)
    sovler_string.test_interval = 100000
    sovler_string.base_lr = base_lr
    sovler_string.lr_policy = 'step'
    sovler_string.gamma = 0.1
    sovler_string.iter_size = 3
    sovler_string.stepsize = 20000
    sovler_string.display = 20
    sovler_string.max_iter = iters
    sovler_string.momentum = 0.9
    sovler_string.weight_decay = 0.0002
    sovler_string.snapshot = 10
    sovler_string.snapshot_prefix = snapshot

    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU

    # print str(sovler_string)
    print(sovler_string)

    # with open('solver.prototxt', 'w') as f:
    #     f.write(str(sovler_string))


def write_network(data_path="../../data/PIOD/Augmentation/train_pair_320x320.lst", batch_size=5):
    n = caffe.NetSpec()
    # n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
    n.data, n.label = L.ImageLabelmapData(include={'phase': 0},  ## 0-TRAIN 1-TEST
                                          image_data_param={
                                              'source': data_path,
                                              'batch_size': batch_size,
                                              'shuffle': True,
                                              'new_height': 0,
                                              'new_width': 0,
                                              'root_folder': "",
                                              'data_type': "h5"},
                                          transform_param={
                                              'mirror': False,
                                              'crop_size': 320,
                                              'mean_value': [104.006988525, 116.668769836, 122.678916931]
                                          },
                                          ntop=2)
    ## Notice: Currently the data layer 'ImageLabelmapData' has some problem !!!!
    ## so when run this func, you also need to manually paste the 'ImageLabelmapData' param into train prototxt

    n.label_edge, n.label_ori = L.Slice(n.label, slice_param={'slice_point': 1}, ntop=2)

    dcnet(n, is_train=True)

    loss_bottoms = [n.unet1b_edge, n.label_edge]
    n.edge_loss = L.ClassBalancedSigmoidCrossEntropyAttentionLoss(*loss_bottoms,
                                                                  loss_weight=1.0,
                                                                  attention_loss_param={'beta': 4.0,
                                                                                        'gamma': 0.5})
    # n.edge_loss = L.ClassBalancedSigmoidCrossEntropyAttentionLoss(*loss_bottoms,
    #                                                               loss_weight=1.0,
    #                                                               attention_loss_param={'beta': 1.0,
    #                                                                                     'gamma': 0}) ## CCE loss

    loss_bottoms = [n.unet1b_ori, n.label_ori, n.label_edge]
    n.ori_loss = L.OrientationSmoothL1Loss(*loss_bottoms, loss_weight=0.5, smooth_l1_loss_param={'sigma': 3.0})


    # loss_bottoms = [n.BCP_crop, n.label_edge]
    # n.edge_loss2 = L.SigmoidCrossEntropyLoss(*loss_bottoms,
    #                                         loss_weight=1.0,
    #                                         attention_loss_param={'beta': 4.0,
    #                                                               'gamma': 0.5})

    with open('dcnet.prototxt', 'w') as f:
        f.write(str(n.to_proto()))  ## write network







def train(initmodel, gpu):
    # write_network(data_path='../../data/BSDSownership/Augmentation/train_pair.lst')
    write_network()
    print('write network in prototxt')
    exit()


    # caffe.set_mode_gpu()
    # caffe.set_device(gpu)
    # write_solver(snapshot='snapshot/dcnet_1')

    # solver = caffe.SGDSolver('solver.prototxt')
    # ## solver.prototxt include the message of train.prototxt (sovler_string.net)
    # if initmodel:
    #     solver.net.copy_from(initmodel)  ## use solver.net.copy_from() to load  pretrained model

    # solver.step(solver.param.max_iter)


if __name__ == '__main__':
    train(initmodel='../dcnet3/ResNet-50-model.caffemodel', gpu=0)

    ## just for test
    # net_forward()
    # write_solver()
