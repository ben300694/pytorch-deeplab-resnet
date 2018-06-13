from __future__ import print_function
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys

# sys.path.insert(0, '/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
# import caffe
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from typing import List

import deeplab_resnet
from collections import OrderedDict
import os
from os import walk
import matplotlib.pyplot as plt
import torch.nn as nn

import yaml

from docopt import docopt

# Load the configuration file
full_path = os.path.realpath(__file__)
config = yaml.safe_load(open(os.path.dirname(full_path) + '/config.yml'))

docstr = """Perform inference of ResNet-DeepLab trained on scenes (VOC 2012), a total of 21 labels including background,
            on images of your choice

Usage: 
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: VOC12_scenes_]
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt(docstr, version='v0.1')
print(args)


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = int(args['--NoLabels']) - 1  # labels from 0,1, ... 20(for VOC)
    count = np.zeros((max_label + 1,))
    for j in range(max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        # pdb.set_trace()
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou


gpu0 = int(args['--gpu0'])
print('Using gpu0 =', gpu0, 'Device name:',torch.cuda.get_device_name(gpu0))
# im_path = args['--testIMpath']
im_path = config['directories']['IMAGE_DIR']

model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))
model.train(False)
model.eval()
model.cuda(gpu0)

snapPrefix = args['--snapPrefix']
gt_path = args['--testGTpath']

img_list = open(config['directories']['lists']['DATA_INFERENCE_LIST_PATH']).readlines()  # type: List[str]
image_format_suffix = '.png'
print(img_list)

MODEL_WEIGHTS = config['RESTORE_FROM']

saved_state_dict = torch.load(MODEL_WEIGHTS)
model.load_state_dict(saved_state_dict)

for i in img_list:
    img = np.zeros((960, 1280, 3))
    img_path = os.path.join(im_path, i[:-1] + image_format_suffix)
    print('Working on ', img_path)

    img_temp = cv2.imread(img_path).astype(float)
    img_original = img_temp
    # Subtract mean from image
    img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
    img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
    img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
    img[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
    # gt groundtruth
    # gt = cv2.imread(os.path.join(gt_path, i[:-1] + image_format_suffix), 0)
    # gt[gt == 255] = 0

    with torch.no_grad():
        input_var = Variable(torch.from_numpy(img[np.newaxis, :].transpose(0, 3, 1, 2)).float()).cuda(gpu0)
        output = model(input_var)

        print(output.size())

    # interp = nn.UpsamplingBilinear2d(size=(513, 513))
    # output = interp(output[3]).cpu().data[0].numpy()
    # output = output[:, :img_temp.shape[0], :img_temp.shape[1]]
    #
    # output = output.transpose(1, 2, 0)
    # output = np.argmax(output, axis=2)
    #
    #
    # plt.subplot(3, 1, 1)
    # plt.imshow(img_original)
    # # plt.subplot(3, 1, 2)
    # # plt.imshow(gt)
    # plt.subplot(3, 1, 3)
    # plt.imshow(output)
    # plt.show()

    # iou_pytorch = get_iou(output, gt)