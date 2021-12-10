# 统计各目标实例数量，以及其中大、中、小目标数量
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import torch
import xml.etree.ElementTree as ET
from lxml import html
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
    area = (ymax - xmax) * (ymin - xmin)  # box面积
    return class_name, area

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
            class_name, area = read_box(obj)
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

    # for key in instance_dict:
    #     if key in class_dict.values():

    print(temp, i, j, k, t, x, y)  # 192424 19025 37213 18120 64269 51761 2075
    # print(instance_list[:, 0].sum(), instance_list[:, 1].sum(), instance_list[:, 2].sum())

    # print(instance_dict)
    # labels, values = zip(*instance_dict.items())
    # # labels = label for label in labels.item()[0:2]
    # # print(labels, labels.shape())
    # plt.bar(labels, values)
    # plt.show()
    # plt.savefig('./results/classes2.jpg')
    # df = pd.DataFrame(instance_list, columns=list('sml'))
    # df.plot(kind='bar', color=['blue', 'orange', 'gray'])
    # plt.show()
    # plt.savefig('./results/classes.jpg')
    # plt.close()

    # print(instance_list)


if __name__ == '__main__':
   main()
































