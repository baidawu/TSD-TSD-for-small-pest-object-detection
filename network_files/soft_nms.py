import torch
import time
import numpy as np
import copy
import random





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

# def IOU(box_a, box_b):
#     inter = intersect(box_a, box_b)
#     area_a = ((box_a[:, 2]-box_a[:, 0]) *
#               (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#     area_b = ((box_b[:, 2]-box_b[:, 0]) *
#               (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#     union = area_a + area_b - inter
#     return inter / union  # [A,B]

def tri_dist(boxes1, boxes2):

    x_center1 = boxes1[:, None, 0] + (boxes1[:, None, 2] - boxes1[:, None, 0]) / 2.
    y_center1 = boxes1[:, None, 1] + (boxes1[:, None, 3] - boxes1[:, None, 1]) / 2.

    x_center2 = boxes2[:, None, 0] + (boxes2[:, None, 2] - boxes2[:, None, 0]) / 2.
    y_center2 = boxes2[:, None, 1] + (boxes2[:, None, 3] - boxes2[:, None, 1]) / 2.
    x_center2 = x_center2.view(1, -1)
    y_center2 = y_center2.view(1, -1)

    dist = torch.sqrt((x_center1 - x_center2) ** 2 + (y_center1 - y_center2) ** 2)

    distance = dist / 35.63
    distance = 1 - distance

    # distance越大越好（类似iou,越大越好）
    return distance


def trid_nms(boxes, scores, thresh):

    _, idx = scores.sort(0, descending=True)  # descending表示降序
    boxes_idx = boxes[idx]
    trid = tri_dist(boxes_idx, boxes_idx).triu_(diagonal=1)  # 取上三角矩阵，不包含对角线
    B = trid
    while 1:
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= thresh).float().unsqueeze(1).expand_as(A)
        B = trid.mul(E)
        if A.equal(B) == True:
            break
    keep = idx[maxA <= thresh]

    return keep

