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
import copy
import random
import cpython_nms
import torchvision
from torchvision.ops import nms

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

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


def nms_cpu_py(dets, box_scores, thre):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = box_scores

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        keep.append(order[0])
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        inter = w * h
        union = area[order[0]] + area[order[1:]] - inter
        iou = inter / union

        inds = np.where(iou <= thre)[0]
        order = order[inds + 1]
    print(len(keep))

    return keep

def cpu_soft_nms(boxes, scores, sigma=0.5, Nt=0.3, threshold=0.05, method=2):
    N = boxes.shape[0]

    for i in range(N):
        # 用冒泡排序法找到分数最高的预测框，并将该预测框放在第i个位置
        maxscore = scores[i]
        maxpos = i

        # 先用一些中间变量存储第i个预测框
        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = scores[i]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < scores[pos]:
                maxscore = scores[pos]
                maxpos = pos
            pos = pos + 1

            # 将分数最高的预测框M放在第i个位置
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        scores[i] = scores[maxpos]

        # 将原先第i个预测框放在分数最高的位置
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        scores[maxpos] = ts

        # 程序到此实现了：寻找第i至第N个预测框中分数最高的框，并将其与第i个预测框互换位置。

        # 预测框M，前缀"t"表示target
        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = scores[i]

        # 下面针对M进行NMS迭代过程，
        # 需要注意的是，如果soft-NMS将score削弱至某阈值threshold以下，则将其删除掉
        # 在程序中体现为，将要删除的框放在了最后，并使 N = N-1
        pos = i + 1
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = scores[pos]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    scores[pos] = weight * scores[pos]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if scores[pos] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        scores[pos, 4] = scores[N - 1]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep

def box_soft_nms(bboxes, scores, labels, nms_threshold=0.5, soft_threshold=0.3, sigma=0.5, mode='union'):
    """
    soft-nms implentation according the soft-nms paper
    :param bboxes: all pred bbox
    :param scores: all pred cls
    :param labels: all detect class label，注：scores只是单纯的得分，需配合label才知道具体对应的是哪个类
    :param nms_threshold: origin nms thres, for judging to reduce the cls score of high IoU pred bbox
    :param soft_threshold: after cls score of high IoU pred bbox been reduced, soft_thres further filtering low score pred bbox
    :return:
    """
    unique_labels = labels.cpu().unique().cuda() # 获取pascal voc 20类标签
    box_keep = []
    labels_keep = []
    scores_keep = []
    for c in unique_labels:  # 相当于NMS中对每一类的操作，对应step-1
        c_boxes = bboxes[labels == c]  # bboxes、scores、labels一一对应，按照label == c就可以取出对应类别 c 的c_boxes、c_scores
        c_scores = scores[labels == c]
        weights = c_scores.clone()
        x1 = c_boxes[:, 0]
        y1 = c_boxes[:, 1]
        x2 = c_boxes[:, 2]
        y2 = c_boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # bbox面积
        _, order = weights.sort(0, descending=True)  # bbox根据score降序排序，对应NMS中step-2
        while order.numel() > 0:  # 对应NMS中step-5
            i = order[0]  # 当前order中的top-1，保存之
            box_keep.append(c_boxes[i])  # 保存bbox
            labels_keep.append(c)  # 保存cls_id
            scores_keep.append(c_scores[i])  # 保存cls_score
            if order.numel() == 1:  # 当前order就这么一个bbox了，那不玩了，下一个类的bbox操作吧
                break
            xx1 = x1[order[1:]].clamp(min=x1[i])  # 别忘了x1[i]对应x1[order[0]]，也即top-1，寻找Inp区域的坐标
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            w = (xx2 - xx1 + 1).clamp(min=0)  # Inp区域的宽、高、面积
            h = (yy2 - yy1 + 1).clamp(min=0)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 经过origin NMS thres，得到高IoU的bboxes index，
            # origin NMS操作就直接剔除掉这些bbox了，soft-NMS就是对这些bbox对应的score做权重降低
            ids_t= (ovr>=nms_threshold).nonzero().squeeze() # 高IoU的bbox，与inds = np.where(ovr >= nms_threshold)[0]功能类似
            weights[[order[ids_t+1]]] *= torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma)
            # soft-nms对高IoU pred bbox的score调整了一次，soft_threshold仅用于对score抑制，score太小就不考虑了
            ids = (weights[order[1:]] >= soft_threshold).nonzero().squeeze() # 这一轮未被抑制的bbox
            if ids.numel() == 0:  # 竟然全被干掉了，下一个类的bbox操作吧
                break
            c_boxes = c_boxes[order[1:]][ids]  # 先取得c_boxes[order[1:]]，再在其基础之上操作[ids]，获得这一轮未被抑制的bbox
            c_scores = weights[order[1:]][ids]
            _, order = c_scores.sort(0, descending=True)
            if c_boxes.dim()==1:
                c_boxes=c_boxes.unsqueeze(0)
                c_scores=c_scores.unsqueeze(0)
            x1 = c_boxes[:, 0]  # 因为bbox已经做了筛选了，areas需要重新计算一遍，抑制的bbox剔除掉
            y1 = c_boxes[:, 1]
            x2 = c_boxes[:, 2]
            y2 = c_boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    return box_keep, labels_keep, scores_keep  # scores_keep保存的是未做权重降低的score，降低权重的score仅用于soft-nms操作

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
    scores = copy.deepcopy(box_scores)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # print(dets[:, 4], scores)

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

        # # linear decay
        alpha = random.random()
        # alpha = random.randint(1, 2)
        weight = torch.ones(ovr.shape).cuda()
        weight[ovr > iou_thresh] = weight[ovr > iou_thresh] - ovr[ovr > iou_thresh] * alpha
        # weight[ovr > iou_thresh] = weight[ovr > iou_thresh] - ovr[ovr > iou_thresh]

        # Gaussian decay
        # ids_over = (ovr > iou_thresh).float()
        # ovr = ovr * ids_over
        # weight = torch.exp(-(ovr * ovr) / sigma)

        # print(alpha)
        scores[pos:] = weight * scores[pos:]  # if ovr[pos:] > iou_thresh else scores[pos:] = scores[pos:]
        # if scores[pos] < thresh:
        #     dets[pos, 0] = dets[N - 1, 0]
        #     dets[pos, 1] = dets[N - 1, 1]
        #     dets[pos, 2] = dets[N - 1, 2]
        #     dets[pos, 3] = dets[N - 1, 3]
        #     scores[pos] = scores[N - 1]
        #     N = N - 1
        #     pos = pos - 1

    # print(dets[:, 4], scores)
    # select the boxes and keep the corresponding indexes
    # keep = dets[:, 4][scores > thresh].int()
    keep = dets[:, 4][scores > thresh].long()
    # print(scores, len(scores), box_scores, len(box_scores))
    # print(box_scores[keep], len(box_scores[keep]))
    # print(len(box_scores[keep]))

    return keep




def speed():
    boxes = 1000 * torch.rand((1000, 1000, 4), dtype=torch.float)
    boxscores = torch.rand((1000, 1000), dtype=torch.float)
    boxscores = boxscores.view(1000, 1000, 1)

    # a = torch.rand(3, 2)
    # b = torch.rand(3)
    # print(a, b, a.size(), b.size())
    # b = b.view(-1, 1)
    # print(a, b, a.size(), b.size())
    # c = torch.cat((a, b), dim=1)
    # print(a, b, c)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    start = time.time()
    for i in range(10):
        # print(boxes[i][:5], boxscores[i][:5])
        # soft_nms_simple(boxes[i], boxscores[i], 0.5, cuda=cuda)
        # keep1 = soft_nms_pytorch(boxes[i], boxscores[i], iou_thresh=0.5, cuda=cuda)
        # keep2 = torch.ops.torchvision.nms(boxes[i], boxscores[i], 0.5)
        print(boxes[i].shape, boxscores[i].shape)
        # boxscores[i] = boxscores[i].view(-1, 1)
        # print(boxes[i].shape, boxscores[i].shape)
        # dets = torch.hstack((boxes[i], boxscores[i]))
        dets = torch.cat((boxes[i], boxscores[i]), dim=1).cpu().numpy().astype(np.float32, copy=False)
        keep = cpython_nms.diounms(dets, 0.5, 1.0)
        print(keep)
        print(boxes[i].shape, boxscores[i].shape)
        # print(len(keep1), len(keep2))
        # print(keep2, len(keep2))

        # torch.ops.torchvision.nms(boxes[i], boxscores[i], 0.5)
        # print(boxes[i][:5], boxscores[i][:5])
        # keep = keep[:5]
        # print(boxes[i][keep], boxscores[i][keep])
        # print(boxes[i][keep], boxscores[i][:5])
        # keep = nms_cpu_py(boxes[i], boxscores[i], thre=0.5)

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

    # keep = soft_nms_simple(boxes, boxscores, cuda=cuda)
    keep = soft_nms_pytorch(boxes, boxscores, iou_thresh=0.5, cuda=cuda)
    # keep = torch.ops.torchvision.nms(boxes, boxscores, 0.7)

    t_end = time_synchronized()
    print("inference+NMS time: {}".format(t_end - t_start))

    print(keep, keep.type())

def IoG(boxes1, boxes2):  # boxes1:box  boxes2:rep_gt

    # A ∩ B / A = A ∩ B / area(A)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    G = box_area(boxes2)

    iog = inter / G
    return iog

def rep_gt(boxes1, boxes2):

    iou = box_iou(boxes1, boxes2)
    top2 = torch.topk(iou, 2)
    keep = top2.indices
    keep = keep[:, 1]


    return boxes2[keep]

def rep_gt_loss(input, target, beta: float = 1. / 9, size_average: bool = True):

    ground_rep = rep_gt(input, target)
    # print(ground_rep)
    # print(input.size(), ground_rep.size())
    iog = IoG(input, ground_rep)
    cond = torch.lt(iog, beta)
    beta = beta + torch.zeros(iog.size())
    # print(iog.size(), beta.size(), beta)
    loss = torch.where(cond, -torch.log(1 - iog), (iog - beta) / (1 - beta) - torch.log(1 - beta))
    if size_average:
        return loss.mean()
    return loss.sum()

if __name__ == '__main__':

    # test()
    # speed()
    boxes1 = 90 * torch.rand((100, 4), dtype=torch.float)
    boxes2 = 100 * torch.rand((100, 4), dtype=torch.float)

    print(rep_gt_loss(boxes1, boxes2))






