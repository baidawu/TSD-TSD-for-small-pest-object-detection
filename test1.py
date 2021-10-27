#
# # coding:utf-8
# import numpy as np
#
# def soft_nms(boxes, sigma=0.5, Nt=0.1, threshold=0.001, method=1):
#     N = boxes.shape[0]
#     pos = 0
#     maxscore = 0
#     maxpos = 0
#
#     for i in range(N):
#         maxscore = boxes[i, 4]
#         maxpos = i
#
#         tx1 = boxes[i,0]
#         ty1 = boxes[i,1]
#         tx2 = boxes[i,2]
#         ty2 = boxes[i,3]
#         ts = boxes[i,4]
#
#         pos = i + 1
#     # get max box
#         while pos < N:
#             if maxscore < boxes[pos, 4]:
#                 maxscore = boxes[pos, 4]
#                 maxpos = pos
#             pos = pos + 1
#
#     # add max box as a detection
#         boxes[i,0] = boxes[maxpos,0]
#         boxes[i,1] = boxes[maxpos,1]
#         boxes[i,2] = boxes[maxpos,2]
#         boxes[i,3] = boxes[maxpos,3]
#         boxes[i,4] = boxes[maxpos,4]
#
#     # swap ith box with position of max box
#         boxes[maxpos,0] = tx1
#         boxes[maxpos,1] = ty1
#         boxes[maxpos,2] = tx2
#         boxes[maxpos,3] = ty2
#         boxes[maxpos,4] = ts
#
#         tx1 = boxes[i,0]
#         ty1 = boxes[i,1]
#         tx2 = boxes[i,2]
#         ty2 = boxes[i,3]
#         ts = boxes[i,4]
#
#         pos = i + 1
#     # NMS iterations, note that N changes if detection boxes fall below threshold
#         while pos < N:
#             x1 = boxes[pos, 0]
#             y1 = boxes[pos, 1]
#             x2 = boxes[pos, 2]
#             y2 = boxes[pos, 3]
#             s = boxes[pos, 4]
#
#             area = (x2 - x1 + 1) * (y2 - y1 + 1)
#             iw = (min(tx2, x2) - max(tx1, x1) + 1)
#             if iw > 0:
#                 ih = (min(ty2, y2) - max(ty1, y1) + 1)
#                 if ih > 0:
#                     ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
#                     ov = iw * ih / ua #iou between max box and detection box
#
#                     if method == 1: # linear
#                         if ov > Nt:
#                             weight = 1 - ov
#                         else:
#                             weight = 1
#                     elif method == 2: # gaussian
#                         weight = np.exp(-(ov * ov)/sigma)
#                     else: # original NMS
#                         if ov > Nt:
#                             weight = 0
#                         else:
#                             weight = 1
#
#                     boxes[pos, 4] = weight*boxes[pos, 4]
#                     print(boxes[:, 4])
#
#             # if box score falls below threshold, discard the box by swapping with last box
#             # update N
#                     if boxes[pos, 4] < threshold:
#                         boxes[pos,0] = boxes[N-1, 0]
#                         boxes[pos,1] = boxes[N-1, 1]
#                         boxes[pos,2] = boxes[N-1, 2]
#                         boxes[pos,3] = boxes[N-1, 3]
#                         boxes[pos,4] = boxes[N-1, 4]
#                         N = N - 1
#                         pos = pos - 1
#
#             pos = pos + 1
#     keep = [i for i in range(N)]
#     return keep
#
#
# if __name__ == '__main__':
#
#     boxes = np.array([[100, 100, 150, 168, 0.63],[166, 70, 312, 190, 0.55],[221, 250, 389, 500, 0.79],[12, 190, 300, 399, 0.9],[28, 130, 134, 302, 0.3]])
#     keep = soft_nms(boxes)
#     print(keep)
# -*- coding:utf-8 -*-
# Author:Richard Fang

import time
import numpy as np
import torch
import torchvision

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

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

def soft_nms_pytorch(dets, box_scores, iou_thresh, sigma=0.5, thresh=0.001, cuda=0):
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
    scores = box_scores
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
        # print(ovr)
        ids_over = (ovr < iou_thresh).float()
        ovr = ovr * ids_over
        # print(ovr)
        # ids_t = (ovr >= iou_thresh).nonzero().squeeze()
        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]  # if ovr[pos:] > iou_thresh else scores[pos:] = scores[pos:]
        # for i in range(pos, N):
        #     if ovr[i] > iou_thresh:
        #         scores[i] = weight[i] * scores[i]


    # select the boxes and keep the corresponding indexes
    # keep = dets[:, 4][scores > thresh].int()
    keep = dets[:, 4][scores > thresh].long()

    return keep


def speed():
    boxes = 1000 * torch.rand((1000, 100, 4), dtype=torch.float)
    boxscores = torch.rand((1000, 100), dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    start = time.time()
    for i in range(1000):
        # soft_nms_simple(boxes[i], boxscores[i], 0.5, cuda=cuda)
        soft_nms_pytorch(boxes[i], boxscores[i], 0.5, cuda=cuda)
        # torch.ops.torchvision.nms(boxes[i], boxscores[i], 0.7)
    end = time.time()
    print("Average run time: %f ms" % (end - start))


def test():
    # boxes and boxscores
    boxes = torch.tensor([[200, 200, 400, 400],
                          [220, 220, 420, 420],
                          [200, 240, 400, 440],
                          [240, 200, 440, 400],
                          [1, 1, 2, 2]], dtype=torch.float)
    boxscores = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.9], dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    t_start = time_synchronized()

    keep = soft_nms_simple(boxes, boxscores, cuda=cuda)
    # keep = torch.ops.torchvision.nms(boxes, boxscores, 0.7)

    t_end = time_synchronized()
    print("inference+NMS time: {}".format(t_end - t_start))

    print(keep,keep.type())


if __name__ == '__main__':

    # test()
    speed()






