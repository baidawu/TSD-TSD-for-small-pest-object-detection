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

def IoU(output, target):

    x1, y1, x2, y2 = output[:, None, 0].view(-1), output[:, None, 1].view(-1), output[:, None, 2].view(-1), output[:, None, 3].view(-1)
    x1g, y1g, x2g, y2g = target[:, None, 0].view(-1), target[:, None, 1].view(-1), target[:, None, 2].view(-1), target[:, None, 3].view(-1)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)


    # inter = (xkis2 - xkis1) * (ykis2 - ykis1)
    # temp = torch.zeros(inter.shape).cuda()
    # cond = (ykis2 > ykis1) & (xkis2 > xkis1)
    # intsctk = torch.where(cond, inter, temp)
    # intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
    w = (xkis2 - xkis1).clamp(min=0)
    h = (ykis2 - ykis1).clamp(min=0)
    intsctk = w * h
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iou = intsctk / unionk

    return iou

def box_iog(output, target):

    x1, y1, x2, y2 = output[:, None, 0].view(-1), output[:, None, 1].view(-1), \
                     output[:, None, 2].view(-1), output[:, None, 3].view(-1)
    x1g, y1g, x2g, y2g = target[:, None, 0].view(-1), target[:, None, 1].view(-1), \
                         target[:, None, 2].view(-1), target[:, None, 3].view(-1)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    temp = torch.zeros(x1.shape).cuda()
    intsctk = torch.where((ykis2 - ykis1 > 0) & (xkis2 - xkis1 > 0), (xkis2 - xkis1) * (ykis2 - ykis1), temp)
    G = (x2g - x1g) * (y2g - y1g) + 1e-7
    iog = intsctk / G

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
    iog = box_iog(input, ground_rep)
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

def smooth_rep_box_loss(input, labels, beta: float = 1. / 9, size_average: bool = False):

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
            mask = (iou > 0)
            # mask = mask.view(-1)
            # boxes_rep = boxes_rep[mask]
            iou = iou[mask]
            loss = iou.sum()
            losses += loss
            length += len(iou)

            # if len(boxes_rep) > 0:
            #     n = torch.abs(boxes[i].view(1, 4) - boxes_rep)
            #     cond = torch.lt(n, beta)
            #     print("rep loss:{}\n{}".format(n, cond))
            #     loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
            #     losses += loss.sum()
            #     length += len(boxes_rep)
            # iou = IoU(boxes[i].view(1, 4), boxes_rep)
            # loss = iou.sum()
            # temp = (iou > 0)
            # temp = iou[temp]
            # length += len(temp)
            # losses += loss
    if length == 0:
        l = (input - input).sum()
        l = l / 1
        return l
    else:
        # print(losses, length, losses / length)
        return losses / length

