import torch
import torch.nn as nn
import numpy as np
import math

def generalized_iou_loss(pr_bboxes, gt_bboxes, beta: float = 1. / 9, reduction='sum'):
    """
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    # TO_REMOVE = beta
    TO_REMOVE = 1.
    gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + TO_REMOVE) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + TO_REMOVE)
    pr_area = (pr_bboxes[:, 2] - pr_bboxes[:, 0] + TO_REMOVE) * (pr_bboxes[:, 3] - pr_bboxes[:, 1] + TO_REMOVE)

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    # wh = rb - lt + TO_REMOVE
    # wh = torch.max(wh, torch.zeros(wh.shape).cuda())
    inter = wh[:, 0].clamp(min=0) * wh[:, 1].clamp(min=0)

    union = gt_area + pr_area - inter
    # union = gt_area + pr_area - inter + TO_REMOVE
    iou = inter / union
    # iou = torch.div(inter, union)
    # enclosure
    clt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    crb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    cwh = (crb - clt + TO_REMOVE).clamp(min=0)
    # cwh = crb - clt + TO_REMOVE
    # cwh = torch.max(cwh, torch.zeros(cwh.shape).cuda())
    enclosure = cwh[:, 0].clamp(min=0) * cwh[:, 1].clamp(min=0)
    # print(union, enclosure)

    giou_term = ((enclosure - union) / enclosure).clamp(min=0)
    # giou_term = torch.div(enclosure - union, enclosure)
    # giou_term = torch.max(giou_term, torch.zeros(giou_term.shape).cuda())
    giou = iou - giou_term
    loss = 1. - giou
    # for i in loss:
    #     print("{}:****".format(i))
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    return loss


def Giou_loss(box_pre, box_gt, beta: float = 1. / 9, reduction='sum'):
    """

    Args:
        box_pre: shape(N, 4) predicted x1, y1, x2, y2
        box_gt: shape(N, 4) groundtruth x1, y1, x2, y2
        method:

    Returns:
        Giou loss
    """

    assert box_pre.shape == box_gt.shape

    temper = 1e-7
    # 预测坐标面积
    box_pre_area = (box_pre[:, 2] - box_pre[:, 0] + 1) * (box_pre[:, 3] - box_pre[:, 1] + 1)
    # 标签坐标面积
    box_gt_area = (box_gt[:, 2] - box_gt[:, 0] + 1) * (box_gt[:, 3] - box_gt[:, 1] + 1)


    # 并集区域坐标
    xx1 = torch.max(box_pre[:, 0], box_gt[:, 0])
    yy1 = torch.max(box_pre[:, 1], box_gt[:, 1])
    xx2 = torch.min(box_pre[:, 2], box_gt[:, 2])
    yy2 = torch.min(box_pre[:, 3], box_gt[:, 3])

    # inter 面积
    w = (xx2 - xx1 + 1).clamp(min=0)
    h = (yy2 - yy1 + 1).clamp(min=0)
    inter = w * h

    union = box_pre_area + box_gt_area - inter
    # iou = inter / union
    temp = torch.zeros(union.shape).cuda()
    iou = torch.where(union != 0., inter / union, temp)

    # 最小封闭形状坐标
    xx1_c = torch.min(box_pre[:, 0], box_gt[:, 0])
    yy1_c = torch.min(box_pre[:, 1], box_gt[:, 1])
    xx2_c = torch.max(box_pre[:, 2], box_gt[:, 2])
    yy2_c = torch.max(box_pre[:, 3], box_gt[:, 3])

    # C面积
    w_c = (xx2_c - xx1_c + 1).clamp(min=0)
    h_c = (yy2_c - yy1_c + 1).clamp(min=0)
    area_c = w_c * h_c

    # print("000{}".format(area_c))  # 为什么<=0?
    # Giou
    # giou_term = (torch.abs(area_c - union) / area_c).clamp(min=0)

    # giou_term = (area_c - union) / area_c
    giou_term = torch.where(area_c != 0., (area_c - union) / area_c, temp)
    giou = iou - 1.0 * giou_term
    # print("111{}\n222{}\n***{}".format(iou, giou_term, giou))
    giou_loss = 1. - giou
    # print("+++{}\n___{}".format(giou_loss, giou_loss.sum()))

    if reduction == 'mean':
        giou_loss = giou_loss.mean()
    elif reduction == 'sum':
        giou_loss = giou_loss.sum()
    elif reduction == 'none':
        pass
    return giou_loss

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


def compute_ciou(output, target, transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    # intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
    temp = torch.zeros(x1.shape).cuda()
    intsctk = torch.where((ykis2 - ykis1 > 0) & (xkis2 - xkis1 > 0), (xkis2 - xkis1) * (ykis2 - ykis1), temp)
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        S = 1 - iouk
        alpha = v / (S + v)
    ciouk = iouk - (u + alpha * v)
    # iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    # iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    # ciouk = ((1 - ciouk) * iou_weights).sum(0) / output.size(0)

    ciou_loss = (1. - ciouk).sum(0)
    # ciou_loss = (1. - ciouk).sum(0) / output.size(0)

    return ciou_loss

def compute_diou(output, target, transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)
    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    temp = torch.zeros(x1.shape).cuda()
    intsctk = torch.where((ykis2 - ykis1 > 0) & (xkis2 - xkis1 > 0), (xkis2 - xkis1) * (ykis2 - ykis1), temp)
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk = iouk - u
    # iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    # iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    # diouk = ((1 - diouk) * iou_weights).sum(0) / output.size(0)
    diou_loss = (1 - diouk).sum(0)

    return diou_loss

def compute_giou(output, target, transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    temp = torch.zeros(x1.shape).cuda()
    intsctk = torch.where((ykis2 - ykis1 > 0) & (xkis2 - xkis1 > 0), (xkis2 - xkis1) * (ykis2 - ykis1), temp)
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    giouk = iouk - ((area_c - unionk) / area_c)
    # iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    # iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    # giouk = ((1 - giouk) * iou_weights).sum(0) / output.size(0)

    giou_loss = (1 - giouk).sum(0)

    return giou_loss