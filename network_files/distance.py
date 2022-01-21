import torch
import numpy as np
import math
import sys

def box_area(boxes):

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def tri_dist(boxes1, boxes2):
    """
    boxes1、boxes2:[N, x1,y1,x2,y2]
    Args:
        boxes1: [N, x1,y1,x2,y2] gt_boxes
        boxes2: [N, x1,y1,x2,y2] proposals

    Returns:
        dist = d_center / (r1 + r2)
        dist越小，证明两框距离越近
    """

    x_center1 = boxes1[:, None, 0] + (boxes1[:, None, 2] - boxes1[:, None, 0]) / 2.
    y_center1 = boxes1[:, None, 1] + (boxes1[:, None, 3] - boxes1[:, None, 1]) / 2.
    # r1 = torch.sqrt((boxes1[:, None, 2] - x_center1) ** 2 + (boxes1[:, None, 3] - y_center1) ** 2)

    x_center2 = boxes2[:, None, 0] + (boxes2[:, None, 2] - boxes2[:, None, 0]) / 2.
    y_center2 = boxes2[:, None, 1] + (boxes2[:, None, 3] - boxes2[:, None, 1]) / 2.
    # r2 = torch.sqrt((boxes2[:, None, 2] - x_center2) ** 2 + (boxes2[:, None, 3] - y_center2) ** 2)

    # print("gt_center:{}{} {}\nproposals_center2:{}{} {}".format(
    #     x_center1, y_center1, x_center1.size(), x_center2, y_center2, x_center2.size()))
    # print("r1:{}\nr2:{}".format(r1, r2))

    x_center2 = x_center2.view(1, -1)
    y_center2 = y_center2.view(1, -1)
    # r2 = r2.view(1, -1)

    dist = torch.sqrt((x_center1 - x_center2) ** 2 + (y_center1 - y_center2) ** 2)
    # print("dist:{} {}".format(dist, dist.size()))

    # distance = dist / (r1 + r2 + 1e-7)
    distance = dist / 35.63
    distance = 1 - distance

    # distance = torch.exp(- dist / 35.63)

    # distance越大越好（类似iou,越大越好）
    return distance

def t_dist(boxes1, boxes2):

    x_center1 = boxes1[:, None, 0] + (boxes1[:, None, 2] - boxes1[:, None, 0]) / 2.
    y_center1 = boxes1[:, None, 1] + (boxes1[:, None, 3] - boxes1[:, None, 1]) / 2.
    r1 = torch.sqrt((boxes1[:, None, 2] - x_center1) ** 2 + (boxes1[:, None, 3] - y_center1) ** 2)

    x_center2 = boxes2[:, None, 0] + (boxes2[:, None, 2] - boxes2[:, None, 0]) / 2.
    y_center2 = boxes2[:, None, 1] + (boxes2[:, None, 3] - boxes2[:, None, 1]) / 2.
    r2 = torch.sqrt((boxes2[:, None, 2] - x_center2) ** 2 + (boxes2[:, None, 3] - y_center2) ** 2)

    x_center2 = x_center2.view(1, -1)
    y_center2 = y_center2.view(1, -1)
    r2 = r2.view(1, -1)

    dist = torch.sqrt((x_center1 - x_center2) ** 2 + (y_center1 - y_center2) ** 2)
    a = torch.pow(r1, 2) + torch.pow(r2, 2) - torch.pow(dist, 2)
    b = 2.0 * r1 * r2

    cond = torch.lt(dist, r1 + r2)
    cond2 = (torch.lt(dist, r1 + r2)) & (torch.lt(r1, dist + r2)) & (torch.lt(r2, dist + r1))

    dist1 = torch.exp(- torch.abs(r1 - r2)) + 1  # 包含
    dist2 = a / (b + 1e-7)  # 相交
    dist3 = torch.exp(- dist / 35.63) - 2  # 相离

    distance = torch.where(cond2, dist2, dist1)
    t_distance = torch.where(cond, distance, dist3)

    # t_distance = t_distance.clamp(min=0)
    # distance = dist / torch.sqrt(avg_area())
    # distance = 1 - distance

    # distance越大越好（类似iou,越大越好）
    return t_distance


def bbox_transform(deltas, weights):
    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    dw = torch.clamp(dw, max=np.log(1000. / 16.))
    dh = torch.clamp(dh, max=np.log(1000. / 16.))

    pred_ctr_x = dx
    pred_ctr_y = dy
    pred_w = torch.exp(dw)
    pred_h = torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)

def tri_loss(output, target, transform_weights=None):

    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    # x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    # x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    # x1, y1, x2, y2 = output[:, None, 0], output[:, None, 1], output[:, None, 2], output[:, None, 3]
    # x1g, y1g, x2g, y2g = target[:, None, 0], target[:, None, 1], target[:, None, 2], target[:, None, 3]
    x1, y1, x2, y2 = output[:, None, 0].view(-1), output[:, None, 1].view(-1), output[:, None, 2].view(-1), output[:, None, 3].view(-1)
    x1g, y1g, x2g, y2g = target[:, None, 0].view(-1), target[:, None, 1].view(-1), target[:, None, 2].view(-1), target[:, None, 3].view(-1)


    # x2 = torch.max(x1, x2)
    # y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2.
    y_p = (y2 + y1) / 2.  # (x_p, y_p)output的center坐标
    x_g = (x1g + x2g) / 2.
    y_g = (y1g + y2g) / 2.  # (x_g, y_g)target的center坐标

    # xkis1 = torch.max(x1, x1g)
    # ykis1 = torch.max(y1, y1g)
    # xkis2 = torch.min(x2, x2g)
    # ykis2 = torch.min(y2, y2g)

    # temp = torch.zeros(x1.shape).cuda()
    # intsctk = torch.where((ykis2 - ykis1 > 0) & (xkis2 - xkis1 > 0), (xkis2 - xkis1) * (ykis2 - ykis1), temp)
    # unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    # iou = intsctk / unionk

    r1 = torch.sqrt((x2 - x_p) ** 2 + (y2 - y_p) ** 2)
    r2 = torch.sqrt((x2g - x_g) ** 2 + (y2g - y_g) ** 2)
    dist = torch.sqrt((x_p - x_g) ** 2 + (y_p - y_g) ** 2)

    cond = torch.lt(dist, r1 + r2)
    cond2 = (torch.lt(dist, r1 + r2)) & (torch.lt(r1, dist + r2)) & (torch.lt(r2, dist + r1))
    # cond3 = torch.le(r1 + r2, dist)

    a = torch.pow(r1, 2) + torch.pow(r2, 2) - torch.pow(dist, 2)
    b = 2.0 * r1 * r2

    # loss1 = 1 - (iou + torch.exp(- torch.abs(r1 - r2)) + 1)  # 包含
    # # loss1 = 0.5 * torch.abs(r1 - r2) ** 2
    # loss2 = 1 - (iou + a / (b + 1e-7))  # 相交
    # loss3 = 1 - (iou + torch.exp(- dist / (r1 + r2 + 1e-7)) - 2)  # 相离
    # # loss3 = dist / (r1 + r2 + 1e-7)

    loss1 = 1 - torch.exp(- torch.abs(r1 - r2))   # 包含
    loss2 = 1 - a / (b + 1e-7)  # 相交
    # loss3 = 2 - iou + torch.exp(- dist / (r1 + r2 + 1e-7))  # 相离
    loss3 = 2 - torch.exp(- dist / 35.63)  # 相离

    loss = torch.where(cond2, loss2, loss1)
    t_loss = torch.where(cond, loss, loss3)


    return t_loss.sum()























