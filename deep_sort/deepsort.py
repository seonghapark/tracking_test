import numpy as np

import torch
import torchvision

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.preprocessing import non_max_suppression

from scipy.stats import multivariate_normal


def get_gaussian_mask():
    #128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma**2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask



class Deepsort_rbc():
    def __init__(self, m_deepsort, width, height, use_cuda):
        self.use_cuda = use_cuda
        self.width = width
        self.height = height

        self.encoder = m_deepsort
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', .5, 100, self.encoder)  ## euclidian or cosine
        self.tracker = Tracker(self.metric, max_iou_distance=0.7, max_age=10, n_init=3)

        if use_cuda:
            self.gaussian_mask = get_gaussian_mask().cuda()
        else:
            self.gaussian_mask = get_gaussian_mask()

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])


    def pre_process(self, frame, boxes):
        crops = []

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])

        width = frame.shape[1]
        height = frame.shape[0]

        for i in range(len(boxes)):
            box = boxes[i]

            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            try:
                crop = frame[y1:y2, x1:x2, :]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)
        return crops


    def a_run_deep_sort(self, frame, boxes):
        if boxes == None or boxes == []:
            self.tracker.predict()
            print('no detections')
            return self.tracker

        if self.use_cuda:
            processed_crops = self.pre_process(frame, boxes).cuda()
        else:
            processed_crops = self.pre_process(frame, boxes)

        processed_crops = self.gaussian_mask * processed_crops

        features = processed_crops
        #features = self.encoder.forward(processed_crops)
        #features = features.detach().cpu().numpy()


        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        detections = []
        out_scores = []
        out_classes = []

        for i in range(len(boxes)):
            detect = boxes[i][:4]
            detect1 = [0]*4
            detect1[0] = detect[0] * self.width   # t
            detect1[1] = detect[1] * self.height  # l
            detect1[2] = (detect[2] - detect[0]) * self.width   # b -> w
            detect1[3] = (detect[3] - detect[1]) * self.height  # r -> h


            detections.append(detect1)
            out_scores.append(boxes[i][5])
            out_classes.append(boxes[i][-1])
        detections = np.asarray(detections)
        out_scores = np.asarray(out_scores)

        dets = [Detection(bbox, score, outclass, feature) for bbox, score, outclass, feature in zip(detections, out_scores, out_classes, features)]


        #outboxes = np.array([d.tlwh for d in dets])
        #outscores = np.array([d.confidence for d in dets])
        #indices = non_max_suppression(outboxes, 0.8, outscores)

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker
