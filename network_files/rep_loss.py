
import copy
import math
import sys
import torch

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


    iou = inter / (area1[:, None] + area2 - inter + 1e-7)
    # print("iou:{}\nboxes1 :{}\nboxes2:{}".format(iou, boxes1, boxes2))

    return iou


def IoG(boxes1, boxes2):  # boxes1:box  boxes2:rep_gt

    # A ∩ B / A = A ∩ B / area(A)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    G = box_area(boxes2) + 1e-7

    iog = inter / G
    return iog


def rep_gt(boxes1, boxes2):
    iou = box_iou(boxes1, boxes2)
    top2 = torch.topk(iou, 2)
    keep = top2.indices
    keep = keep[:, 1]


    return boxes2[keep]


def rep_gt_loss(input, target, beta: float = 1. / 9, size_average: bool = False):

    ground_rep = rep_gt(input, target)
    # print(ground_rep)
    # print(input.size(), ground_rep.size())
    iog = IoG(input, ground_rep)
    # cond = torch.lt(iog, beta)
    # beta = beta + torch.zeros(iog.size()).cuda()
    # loss = torch.where(cond, -torch.log(1 - iog), (iog - beta) / (1 - beta) - torch.log(1 - beta))
    # print(loss)

    beta = beta + torch.zeros(iog.size()).cuda()
    cond = torch.lt(beta, iog)
    # print(iog.size(), beta.size(), beta)
    loss = -torch.log(1 - iog)
    # loss = torch.where(cond,  (iog - beta) / (1 - beta) - torch.log(1 - beta), -torch.log(1 - iog))

    if size_average:
        return loss.mean()
    return loss.sum()

def smooth_rep_loss(input, target, beta: float = 1. / 9, size_average: bool = False):

    ground_rep = rep_gt(input, target)
    n = torch.abs(input - ground_rep)
    # print(torch.max(n), torch.exp(0.5 * torch.max(n) ** 2 / beta) - 1)
    # print(n, torch.exp(0.5 * n ** 2 / beta) - 1)
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    # loss = torch.where(cond, 0.5 * n ** 2, n - 0.5)
    if size_average:
        return loss.mean()
    return loss.sum()

def smooth_rep_box_loss(input, labels):

    # boxes_rep, length = rep_box(input, labels)
    boxes = input
    losses = 0.
    length = 0
    # print(input.size(), labels.size())
    for i in range(len(boxes)):
        label = labels[i]
        keep = (labels != label)
        # print(keep, keep.size())
        keep = keep.view(-1)
        # print(keep, keep.size())
        # print(input.size(), keep, keep.size())
        boxes_rep = boxes[keep]
        # print(boxes_rep, boxes_rep.size())

        # print("~~~{}~~{}".format(boxes[i].view(1, 4), boxes_rep))
        if len(boxes_rep) > 0:
            # if len(boxes_rep) == 1:
            #     print(boxes_rep)
            iou = box_iou(boxes[i].view(1, 4), boxes_rep)
            loss = iou.sum()
            # if not math.isfinite(loss):  # 当计算的损失为无穷大时停止训练
            #     print("rep_box_loss is nan, stop training!")
            #     print("label:{}\nkeep:{}".format(label, keep))
            #     print("boxes:{}\ninput:{}".format(boxes, input))
            #     print("boxes[i]:{}\nboxes_rep:{}\niou:{}".format(boxes[i].view(1, 4), boxes_rep, iou))
            #     sys.exit(1)

            temp = (iou > 0)
            temp = iou[temp]
            length += len(temp)
            losses += loss
            # print("loss:{} len loss:{} losses:{}".format(loss, length, losses))
            # if len(loss) > 0:
            #     length = length + len(loss)
            #     loss = loss.sum()
            #     # print("--{}".format(loss))
            #     losses = losses + loss
                # print("++{}".format(losses))

    if length == 0:

        # print("losses:{} loss:{}".format(losses, torch.tensor(0.0, device="cuda:0",  requires_grad=True)))
        l = (input - input).sum()
        l = l / 1
        # print(l)
        # print(torch.zeros(1).cuda())
        return l
    else:
        # print(losses, length, losses / length)
        return losses / length

