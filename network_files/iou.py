import torch
import torch.nn as nn
import numpy as np
import math

def box_area(boxes):

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

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

def bbox_ciou(boxes1, boxes2, transform_weights=None):

    w_pred = torch.abs(boxes1[:, None, 1] - boxes1[:, None, 0])
    h_pred = torch.abs(boxes1[:, None, 3] - boxes1[:, None, 2])
    w_gt = torch.abs(boxes2[:, None, 1] - boxes2[:, None, 0])
    h_gt = torch.abs(boxes2[:, None, 3] - boxes2[:, None, 2])

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    ltc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    x_center = (boxes1[:, None, 1] + boxes1[:, None, 0]) / 2
    y_center = (boxes1[:, None, 3] + boxes1[:, None, 2]) / 2
    x_center_g = (boxes1[:, None, 1] + boxes1[:, None, 0]) / 2
    y_center_g = (boxes1[:, None, 3] + boxes1[:, None, 2]) / 2


    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)

    whc = ((rbc - ltc) ** 2).clamp(min=0)
    c = whc[:, :, 0] * whc[:, :, 1] + 1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    # v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    v = (4 / (math.pi ** 2)) * ((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)) ** 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)
    ciou = iou - (u + alpha * v)

    return ciou


def bbox_diou(boxes1, boxes2, transform_weights=None):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    ltc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    x_center = (boxes1[:, None, 1] + boxes1[:, None, 0]) / 2
    y_center = (boxes1[:, None, 3] + boxes1[:, None, 2]) / 2
    x_center_g = (boxes1[:, None, 1] + boxes1[:, None, 0]) / 2
    y_center_g = (boxes1[:, None, 3] + boxes1[:, None, 2]) / 2

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)

    whc = torch.pow((rbc - ltc), 2).clamp(min=0)
    c = whc[:, :, 0] * whc[:, :, 1] + 1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    diou = iou - u

    return diou


def bbox_giou(boxes1, boxes2, transform_weights=None):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    ltc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = (area1[:, None] + area2 - inter)
    iou = inter / union

    whc = (rbc - ltc).clamp(min=0)
    area_c = whc[:, :, 0] * whc[:, :, 1] + 1e-7
    giou = iou - ((area_c - union) / area_c)

    return giou