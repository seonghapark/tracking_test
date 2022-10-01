# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import sys
import time
import random
import urllib
import numpy as np
import cv2

import torch
from torchvision import transforms

from vicreg import VICReg

from yolov4.utils import *
from yolov4.torch_utils import do_detect
from yolov4.darknet2pytorch import Darknet

from deep_sort.deepsort import Deepsort_rbc

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Data
    parser.add_argument("--input-video", type=str, required=True, help="path to dataset")
    parser.add_argument('--labels', dest='labels',
                        action='store', default='yolov4/coco.names', type=str,
                        help='Labels for detection')
    parser.add_argument('--conf-thresh', type=float, default=0.4)

    # Checkpoint
    parser.add_argument("--vicreg-pretrained", type=Path, required=True, help="path to pretrained model")
    parser.add_argument("--resnet-pretrained", type=Path, required=True, help="path to pretrained model")
    parser.add_argument("--exp-dir", required=True, type=Path,
                        metavar="DIR", help="path to checkpoint directory")

    return parser.parse_args()


class VICReg_Main():
    def __init__(self, args):
        self.model = VICReg(args).cuda()
        self.model.cuda()
        self.model.eval()


    def run(self, args, x, y):
        start_time = time.time()
        # evaluate

        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        x = transform(x).unsqueeze(0)
        y = transform(y).unsqueeze(0)

        x = x.cuda(self.gpu, non_blocking=True)
        y = y.cuda(self.gpu, non_blocking=True)

        loss, repr_loss, std_loss, cov_loss = self.model.forward(x, y)
        #print('repr_loss', (args.sim_coeff * repr_loss * 100).item(),
        #      'std_loss', (args.std_coeff * std_loss).item(),
        #      'cov_loss', (args.cov_coeff * cov_loss).item())

        return (args.sim_coeff * repr_loss * 1000).item()


class YOLOv4_Main():
    def __init__(self, args, cfgfile='yolov4.cfg', weightfile='yolov4.weights'):
        self.use_cuda = True
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = 0.6

        self.m = Darknet(cfgfile)
        self.m.load_weights(weightfile)
        self.m.cuda().eval()

        self.class_names = load_class_names(args.labels)

    def run(self, frame):
        sized = cv2.resize(frame, (512, 512))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(self.m, sized, self.conf_thresh, self.nms_thresh, self.use_cuda)

        return boxes[0]


if __name__ == "__main__":
    args = get_arguments()

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    cap = cv2.VideoCapture(args.input_video)
    ret, frame = cap.read()
    h, w, c = frame.shape
    print(h, w, c)

    yolov4_main = YOLOv4_Main(args)
    vicreg_main = VICReg_Main(args)
    dsort = Deepsort_rbc(vicreg_main.model, w, h, use_cuda=True)

    tracklet = []
    prev_tracklet = []
    c = 0
    while True:
        c += 1
        print(c)
        ret, frame = cap.read()
        if ret == False:
            print('no video frame')
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = yolov4_main.run(frame)
        tracker = dsort.a_run_deep_sort(frame, result)
