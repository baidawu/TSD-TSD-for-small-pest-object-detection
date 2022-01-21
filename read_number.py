# 统计各目标实例数量，以及其中大、中、小目标数量
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from lxml import html
import numpy as np
etree = html.etree

# data_classes = {
#     "1": "Rice planthopper",
#     "15": "Spodoptera cabbage",
#     "2": "Rice Leaf Roller",
#     "16": "Scotogramma trifolii Rottemberg",
#     "3": "Chilo suppressalis",
#     "24": "Yellow tiger",
#     "5": "Armyworm",
#     "25": "Land tiger",
# 	"6": "Bollworm",
#     "28": "Eight-character tiger",
# 	"7": "Meadow borer",
#     "29": "Holotrichia oblita",
# 	"8": "Athetis lepigone",
#     "31": "Holotrichia parallela",
# 	"10": "Spodoptera litura",
#     "32": "Anomala corpulenta",
#     "11": "Spodoptera exigua",
#     "34": "Gryllotalpa orientalis",
# 	"12": "Stem borer",
#     "35": "Nematode trench",
# 	"13": "Little Gecko",
#     "36": "Agriotes fuscicollis Miwa",
# 	"14": "Plutella xylostella",
#     "37": "Melahotus"
# }

def draw_data():
   print("0")



def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def read_box(obj):
    xmin = float(obj["bndbox"]["xmin"])  # 0
    xmax = float(obj["bndbox"]["xmax"])  # 1
    ymin = float(obj["bndbox"]["ymin"])  # 2
    ymax = float(obj["bndbox"]["ymax"])  # 3

    class_name = int(obj["name"])
    # area = (ymax - ymin) * (xmax - xmin)  # box面积
    h = ymax - ymin
    w = xmax - xmin
    area = h * w
    return class_name, area, h / w


def main():
    # read class_indict 解析json文件
    json_file = './pascal_voc_classes.json'
    assert os.path.exists(json_file), "{} file not exist.".format(json_file)
    json_file = open(json_file, 'r')
    class_dict = json.load(json_file)  # class_dict存储所有类别名称和index：index：value
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    xml_root = "./Pest24/Annotations"
    instance_list = []
    instance_dict = {}
    i, j, k, t, x, y = 0, 0, 0, 0, 0, 0
    h_w1, h_w2, h_w3 = 0, 0, 0
    for i in range(38):
        instance_list.append([])
        for j in range(3):
            instance_list[i].append(0)
    temp = 0
    for file in os.listdir(xml_root):
        if not file.endswith('xml'):
            continue
        xml_path = os.path.join(xml_root, file)
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = parse_xml_to_dict(xml)["annotation"]  # 读取的标注数据

        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            temp += 1
            class_name, area, h_w = read_box(obj)
            # h_w = read_h_w(obj)
            # class_name = category_index[class_name].upper()
            # class_name = class_name[0:2].upper()
            # if class_name in instance_dict.keys():
            #     instance_dict[class_name] = instance_dict[class_name] + 1
            # else:
            #     # instance_dict.append({value: '1'})
            #     instance_dict[class_name] = 1
            if area <= 32 ** 2:  # small
                # instance_list[class_name][0] += 1
                i += 1
            elif 32 ** 2 < area <= 96 ** 2:  # medium
                # instance_list[class_name][1] += 1
                j += 1
            elif 96 ** 2 < area <= 128 ** 2:  # large
                # instance_list[class_name][2] += 1
                k += 1
            elif 128 ** 2 < area <= 256 ** 2:
                t += 1
            elif 256 ** 2 < area <= 512 ** 2:
                x += 1
            elif area > 512 ** 2:
                y += 1

            if h_w <= 1:
                h_w1 += 1
            elif 1 < h_w <= 2:
                h_w2 += 1
            elif h_w > 2:
                h_w3 += 1

    #192424 101162 91291 10 0 0 0
    print(temp, i, j, k, t, x, y)  # 192424 19025 37213 18120 64269 51761 2075

    print(h_w1, h_w2, h_w3)


def avg_area():

    xml_root = "./Pest24/Annotations"
    areas = []
    for file in os.listdir(xml_root):
        if not file.endswith('xml'):
            continue
        xml_path = os.path.join(xml_root, file)
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = parse_xml_to_dict(xml)["annotation"]  # 读取的标注数据
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            class_name, area, h_w = read_box(obj)
            areas.append(area)

    sum_area = 0.0
    for i in areas:
        sum_area += i

    avg = sum_area / len(areas)
    # 192424 1269.7345809254564 35.63333524840829
    print(len(areas), avg, avg ** 0.5)

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
    # print(boxes1.size(), boxes2.size(), iou.size())
    return iou

def sqrt_area():
    boxes = torch.tensor([[0, 0, 16, 16],
                          [0, 0, 32, 32],
                          [0, 0, 64, 64],
                          [0, 0, 96, 96]], dtype=torch.float)
    area = box_area(boxes)

    return torch.sqrt(area.sum() / len(boxes))


def tri_dist(boxes1, boxes2):

    x_center1 = boxes1[:, None, 0] + (boxes1[:, None, 2] - boxes1[:, None, 0]) / 2.
    y_center1 = boxes1[:, None, 1] + (boxes1[:, None, 3] - boxes1[:, None, 1]) / 2.
    # r1 = torch.sqrt((boxes1[:, None, 2] - x_center1) ** 2 + (boxes1[:, None, 3] - y_center1) ** 2)

    x_center2 = boxes2[:, None, 0] + (boxes2[:, None, 2] - boxes2[:, None, 0]) / 2.
    y_center2 = boxes2[:, None, 1] + (boxes2[:, None, 3] - boxes2[:, None, 1]) / 2.
    # r2 = torch.sqrt((boxes2[:, None, 2] - x_center2) ** 2 + (boxes2[:, None, 3] - y_center2) ** 2)

    x_center2 = x_center2.view(1, -1)
    y_center2 = y_center2.view(1, -1)
    # r2 = r2.view(1, -1)

    dist = torch.sqrt((x_center1 - x_center2) ** 2 + (y_center1 - y_center2) ** 2)
    # a = torch.pow(r1, 2) + torch.pow(r2, 2) - torch.pow(dist, 2)
    # b = 2.0 * r1 * r2
    #
    # cond = torch.lt(dist, r1 + r2)
    # cond2 = (torch.lt(dist, r1 + r2)) & (torch.lt(r1, dist + r2)) & (torch.lt(r2, dist + r1))
    #
    # dist1 = torch.exp(- torch.abs(r1 - r2)) + 1  # 包含
    # dist2 = a / (b + 1e-7)  # 相交
    # dist3 = torch.exp(- dist / sqrt_area()) - 2  # 相离
    #
    # distance = torch.where(cond2, dist2, dist1)
    # t_distance = torch.where(cond, distance, dist3)

    # t_distance = t_distance.clamp(min=0)
    distance = dist / sqrt_area()
    distance = 1 - distance
    # distance = distance.clamp(min=0)

    # distance越大越好（类似iou,越大越好）
    return distance



def deviation(is_iou=True, is_large=True):
    boxes = torch.tensor([[0, 0, 16, 16],
                          [0, 0, 32, 32],
                          [0, 0, 64, 64],
                          [0, 0, 96, 96]], dtype=torch.float)
    dists = []
    for i in range(0, len(boxes)):
        boxA = boxes[i]
        w, h = boxA[2] - boxA[0], boxA[3] - boxA[1]
        boxB = []
        if is_large:
            for j in range(-15, 16):
                box = [3 * j, 3 * j, 3 * j + w, 3 * j + h]
                boxB.append(box)
        else:
            for j in range(-15, 16):
                if j >= 0:
                    box = [3 * j, 3 * j, 3 * j + w / 2, 3 * j + h / 2]
                else:
                    box = [2 * j + w / 2, 2 * j + h / 2, 2 * j + w, 2 * j + h]
                boxB.append(box)
        boxB = torch.tensor(boxB, dtype=torch.float)
        boxA = boxA.view(-1, 4)

        if is_iou:
            dist = box_iou(boxA, boxB)
        else:
            dist = tri_dist(boxA, boxB)
        dists.append(dist)

    print(dists)

    return dists

def deviation_hor(is_iou=True, is_large=True):
    boxes = torch.tensor([[0, 0, 16, 16],
                          [0, 0, 32, 32],
                          [0, 0, 64, 64],
                          [0, 0, 96, 96]], dtype=torch.float)
    dists = []
    for i in range(0, len(boxes)):
        boxA = boxes[i]
        w, h = boxA[2] - boxA[0], boxA[3] - boxA[1]
        boxB = []
        if is_large:
            for j in range(-15, 16):
                if j >= 0:
                    box = [3 * j, 0, 3 * j + w, h]
                else:
                    box = [3 * j, 0, 3 * j + w, h]
                boxB.append(box)
        # else:
        #     for j in range(-15, 16):
        #         if j >= 0:
        #             box = [2 * j, 2 * j, 2 * j + w / 2, 2 * j + h / 2]
        #         else:
        #             box = [2 * j + w / 2, 2 * j + h / 2, 2 * j + w, 2 * j + h]
        #         boxB.append(box)
        boxB = torch.tensor(boxB, dtype=torch.float)
        boxA = boxA.view(-1, 4)

        if is_iou:
            dist = box_iou(boxA, boxB)
        else:
            dist = tri_dist(boxA, boxB)
        dists.append(dist)

    print(dists)

    return dists

def plot_curve(values, y_label, name, is_iou=True):

    # fig = plt.figure(1)
    try:
        plt.figure()
        colors = ['r', 'g', 'y', 'b']
        x = []
        for i in range(-15, 16):
            x.append(3 * i)
        x = np.array(x)
        for i in range(0, 4):
            value = values[i]
            value = value[0].numpy()
            plt.plot(x, value, colors[i], label='{}'.format(2 ** (4 + i)))
            plt.scatter(x, value, c=colors[i])

            plt.xlabel('Deviation')
            plt.ylabel(y_label)
            plt.legend()

        # 设置横轴精准刻度
        plt.xticks(np.arange(-50, 60, step=10))
        # 设置纵轴精准刻度
        if is_iou:
            plt.yticks(np.arange(0, 1.2, step=0.2))
        # else:

        plt.title('{}-Deviation Curve'.format(y_label))
        plt.savefig('./results/deviation_{}.jpg'.format(name))
        plt.show()
        plt.close()
    except Exception as e:
        print(e)



if __name__ == '__main__':
    # avg_area()
    iou_l = deviation(is_iou=True, is_large=True)
    plot_curve(iou_l, 'IoU', 'iou_l')
    dist_l = deviation(is_iou=False, is_large=True)
    plot_curve(dist_l, 'TriD', 'trid_l', is_iou=False)
    iou_s = deviation(is_iou=True, is_large=False)
    plot_curve(iou_s, 'IoU', 'iou_s')
    dist_s = deviation(is_iou=False, is_large=False)
    plot_curve(dist_s, 'TriD', 'trid_s', is_iou=False)

    iou_h = deviation_hor(is_iou=True, is_large=True)
    plot_curve(iou_h, 'IoU', 'iou_hor_l')
    dist_h = deviation_hor(is_iou=False, is_large=True)
    plot_curve(dist_h, 'TriD', 'trid_hor_l', is_iou=False)


































