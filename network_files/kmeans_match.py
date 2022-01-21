'''
    计算每个anchor最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors，背景-1，废弃-2
'''
import numpy as np
import copy
# 第二步：使用IOU度量，将每个box分配给与其距离最近的anchor
import torch


def box_iou_wh(box1, box2):
    w1, h1 = box1[2:4]
    w2, h2 = box2[2:4]
    s1 = w1*h1
    s2 = w2*h2
    intersection = min(h1,h2) * min(w1,w2)
    if ((w1 < w2) and (h1 < h2)) or ((w1 > w2) and (h1 > h2)):
        union = max(w1,w2) * max(h1,h2)
    else:
        union = s1 + s2 - intersection
    iou = intersection / union
    return iou

# 开始分簇，求均值，更新anchors
def kmeans(anchors, boxes, anchors_num):
    loss = 0
    groups = []
    new_anchors = []
    # 创建2个聚类 2类，正样本和负样本
    for i in range(anchors_num):
        groups.append([])
    # 遍历每个框
    for box in boxes:
        ious = []
        # 遍历每个初始聚类中心anchor，计算当前box与每个中心的iou，找出最大的IOU后将当前box归为对应的类
        for anchor in anchors:
            iou = box_iou_wh(box, anchor)
            ious.append(iou)
        index_of_maxiou = ious.index(max(ious))
        groups[index_of_maxiou].append(box)

    # 求每个聚类中，框的w, h 的均值
    for group in groups:
        w_sum = 0
        h_sum = 0
        length = len(group)
        for box_in_group in group:
            w_sum += box_in_group[2]
            h_sum += box_in_group[3]
        w_mean = w_sum / length
        h_mean = h_sum / length
        # 计算iou时并不关心xy， 所以这里xy设置为默认0
        anchor = np.array([0,0,w_mean,h_mean])
        new_anchors.append(anchor)
    return new_anchors


# 第三步：重复调用kmean函数，直到满足要求：Ⅰ循环次数结束，或者Ⅱ平均值不再变化（代表找到了该类的中心）
def do_kmeans(anchors, boxes, anchors_num, cycle_num):
    cycle = 0
    new_anchors = kmeans(anchors, boxes, anchors_num)
    while True:
        final_anchors = new_anchors
        new_anchors = kmeans(new_anchors, boxes, anchors_num)
        # for anchor in new_anchors:
        # loss = final_anchors -
        cycle += 1
        # if cycle % 10 == 0:
        #     print('循环了%d次'%(cycle))
        flag = np.zeros((9))
        for i in range(len(final_anchors)):
            equal = (new_anchors[i] == final_anchors[i]).all()
            flag[i] = equal
        if flag.all():
            print('循环了', cycle, '次，终于找到了中心anchors')
            break
        if cycle == cycle_num:
            print('循环次数使用完毕')
            break
    # 截取 w ，h
    final_anchors = [anchor[2:4].astype('int32') for anchor in final_anchors ]
    #由小到大排序
    final_anchors = sorted(final_anchors, key=lambda anchor: anchor[0])
    #换成YOLOV3算法中需要的形式，即变成一个列表[w,h,w,h...w,h,w,h]
    true_final_anchors = []
    for anchor in final_anchors:
        true_final_anchors.append(anchor[0])
        true_final_anchors.append(anchor[1])
    return true_final_anchors


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

def kmeans_nms(boxes, box_scores, labels, nms_thresh, top_k):


    boxes = copy.deepcopy(boxes)
    labels = copy.deepcopy(labels)
    scores = copy.deepcopy(box_scores)

    while True:
        max_scores, max_pos = torch.max(scores, dim=0)
        label = labels[max_pos]  # 最大值的label
        keep = labels == label
        ious = box_iou(boxes, boxes[keep])
        mask = ious > nms_thresh
        if len(boxes) < 1:
            break


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





























