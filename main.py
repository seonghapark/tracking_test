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
    parser.add_argument("--exp-dir", required=True, type=Path,
                        metavar="DIR", help="path to checkpoint directory")

    return parser.parse_args()


class VICReg_Main():
    def __init__(self, args):
        args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node
        self.gpu = 0
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=0
        )

        torch.cuda.set_device(self.gpu)
        torch.backends.cudnn.benchmark = True

        self.model = VICReg(args).cuda(self.gpu)
        self.model.cuda(self.gpu)
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
    yolov4_main = YOLOv4_Main(args)
    vicreg_main = VICReg_Main(args)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    cap = cv2.VideoCapture(args.input_video)
    tracklet = []
    prev_tracklet = []
    while True:
        ret, frame = cap.read()
        height, width, channels = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == False:
            print('no video frame')
            break

        result = yolov4_main.run(frame)

        print('result')
        for i in result:
            if i[-1] == 2 or i[-1] == 5 or i[-1] == 7:
                print(i)

        print('new')
        tracklet = []
        for i in result:
            if i[-1] == 2 or i[-1] == 5 or i[-1] == 7:
                x1 = int(i[0] * width)
                y1 = int(i[1] * height)
                x2 = int(i[2] * width)
                y2 = int(i[3] * height)

                h = y2 - y1
                w = x2 - x1

                crop_image = frame[y1:y1+h, x1:x1+w]
                tracklet.append(crop_image)

        if tracklet != []:
            for i in range(len(prev_tracklet)):
                print('new i')
                for j in range(len(tracklet)-1, -1, -1):
                    if i <= j:
                        input1 = prev_tracklet[i]
                        input2 = tracklet[j]
                        repr_loss = vicreg_main.run(args, input1, input2)
                        print(i, j, repr_loss)
            prev_tracklet = tracklet

        time.sleep(2)
