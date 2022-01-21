import os
import json

from lxml import html
etree = html.etree


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

    # class_name = int(obj["name"])
    # # area = (ymax - ymin) * (xmax - xmin)  # box面积
    h = ymax - ymin
    w = xmax - xmin
    area = h * w
    return area


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
            area = read_box(obj)
            areas.append(area)


    print(areas.sum() / len(areas))

    # return areas.sum() / len(areas)
if __name__ == '__main__':
    avg_area()































