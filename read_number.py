# 统计各目标实例数量，以及其中大、中、小目标数量
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import xml.etree.ElementTree as ET
from lxml import html
etree = html.etree

data_classes = {
    "1": "Rice planthopper",
    "15": "Spodoptera cabbage",
    "2": "Rice Leaf Roller",
    "16": "Scotogramma trifolii Rottemberg",
    "3": "Chilo suppressalis",
    "24": "Yellow tiger",
    "5": "Armyworm",
    "25": "Land tiger",
	"6": "Bollworm",
    "28": "Eight-character tiger",
	"7": "Meadow borer",
    "29": "Holotrichia oblita",
	"8": "Athetis lepigone",
    "31": "Holotrichia parallela",
	"10": "Spodoptera litura",
    "32": "Anomala corpulenta",
    "11": "Spodoptera exigua",
    "34": "Gryllotalpa orientalis",
	"12": "Stem borer",
    "35": "Nematode trench",
	"13": "Little Gecko",
    "36": "Agriotes fuscicollis Miwa",
	"14": "Plutella xylostella",
    "37": "Melahotus"
}

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


if __name__ == '__main__':
    xml_root = "./Pest24/Annotations"
    instance_list = []

    for i in range(38):
        instance_list.append([])
        for j in range(3):
            instance_list[i].append(0)

    for file in os.listdir(xml_root):
        if not file.endswith('xml'):
            continue
        xml_path = os.path.join(xml_root,file)
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = parse_xml_to_dict(xml)["annotation"]  # 读取的标注数据

        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            class_name, area = read_box(obj)
            if area <= 32**2:  # small
                instance_list[class_name][0] += 1
            elif 32**2 < area <= 96**2:  # medium
                instance_list[class_name][1] += 1
            elif area > 96**2:  # large
                instance_list[class_name][2] += 1

    df = pd.DataFrame(instance_list,columns=list('sml'))
    df.plot(kind='bar',color=['blue', 'orange', 'gray'])
    plt.show()
    plt.savefig('./results/classes.jpg')
    plt.close()

    print(instance_list)

































