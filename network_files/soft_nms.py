import torch
import time
import numpy as np
import copy
import random

import numpy as np




def soft_nms_pytorch(dets, box_scores, iou_thresh, sigma=0.5, thresh=0.1, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    # scores = box_scores
    # 原boxscores值不发生改变
    scores = copy.deepcopy(box_scores)


    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        # yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        # # yy1 = max(dets[i,1],dets)
        # xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        # yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        # xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        #
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        # inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)

        yy1 = torch.max(dets[i, 0], dets[pos:, 0])
        xx1 = torch.max(dets[i, 1], dets[pos:, 1])
        yy2 = torch.min(dets[i, 2], dets[pos:, 2])
        xx2 = torch.min(dets[i, 3], dets[pos:, 3])
        # print(yy1, xx1, yy2, xx2)
        w1 = xx2 - xx1 + 1
        h1 = yy2 - yy1 + 1
        # print(w1, h1)
        w = torch.max(w1, torch.zeros(w1.shape).cuda())
        h = torch.max(h1, torch.zeros(h1.shape).cuda())
        # print(w, h)

        inter = (w * h).clone().detach().cuda() if cuda else (w * h).clone().detach()
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # # traditional nms
        # weight = torch.ones(ovr.shape).cuda()
        # weight[ovr > iou_thresh] = 0

        # linear decay
        alpha = random.random()
        # alpha = random.randint(1, 2)
        weight = torch.ones(ovr.shape).cuda()
        weight[ovr > iou_thresh] = weight[ovr > iou_thresh] - ovr[ovr > iou_thresh] * alpha
        # weight[ovr > iou_thresh] = weight[ovr > iou_thresh] - ovr[ovr > iou_thresh]

        # # Gaussian decay
        # ids_over = (ovr > iou_thresh).float()
        # ovr = ovr * ids_over
        # weight = torch.exp(-(ovr * ovr) / sigma)

        scores[pos:] = weight * scores[pos:]


    # select the boxes and keep the corresponding indexes
    # keep = dets[:, 4][scores > thresh].int()
    keep = dets[:, 4][scores > thresh].long()

    return keep

def box_area(box):

    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    area1 = box_area(box1)
    area2 = box_area(box2)
    # print(box1)
    # print(box1[:2], box1[2:])

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(box1[:2], box2[:2])
    rb = torch.min(box1[2:], box2[2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[0] * wh[1]

    iou = inter / (area1 + area2 - inter)
    return iou


def soft_nms_simple(dets, box_scores, iou_thresh, sigma=0.5, thresh=0.5, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    # y1 = dets[:, 0]
    # x1 = dets[:, 1]
    # y2 = dets[:, 2]
    # x2 = dets[:, 3]
    scores = box_scores
    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 面积
    # print(dets)

    maxscore, maxpos = torch.max(scores[0:], dim=0)  # 分数最大值、最大值位置

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        if tscore < maxscore:
            ovr = box_iou(dets[i][:-1], dets[maxpos][:-1])
            if ovr > iou_thresh:
                weight = torch.exp(-(ovr * ovr) / sigma)
                scores[i] = weight * scores[i]

    # print(scores)
    keep = dets[:, 4][scores > thresh].long()
    # print(keep)

    return keep

