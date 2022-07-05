import json
import os
import cv2
import torch
import matplotlib.pyplot as plt
import time
from PIL import Image
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box
from lxml import html
etree = html.etree

def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
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

    return xmin, xmax, ymin, ymax

def plt_annotation():
    json_file = './pascal_voc_classes.json'
    assert os.path.exists(json_file), "{} file not exist.".format(json_file)
    json_file = open(json_file, 'r')
    class_dict = json.load(json_file)  # class_dict存储所有类别名称和index：index：value
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    xml_root = "./Pest24/Annotations"
    files = os.listdir(xml_root)
    files = files[10:22]

    for file in files:
        if not file.endswith('xml'):
            continue
        xml_path = os.path.join(xml_root, file)
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = parse_xml_to_dict(xml)["annotation"]  # 读取的标注数据

        assert "object" in data, "{} lack of object information.".format(xml_path)
        name = data['filename']
        img_path = './Pest24/JPEGImages/{}.jpg'.format(name)
        img = cv2.imread(img_path)
        for obj in data["object"]:
            # xmin, xmax, ymin, ymax = read_box(obj)
            xmin = int(obj["bndbox"]["xmin"])  # 0
            xmax = int(obj["bndbox"]["xmax"])  # 1
            ymin = int(obj["bndbox"]["ymin"])  # 2
            ymax = int(obj["bndbox"]["ymax"])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imwrite('./results/annotation/gt_{}.jpg'.format(name), img)



def create_model(num_classes):

    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model

def read_predict():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=25) # 24类+背景

    # load train weights
    train_weights = "./save_weights_ours/resNetFpn-model-12-20220331-084739.pth"
    #train_weights = "./save_weights/"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    state_dict = torch.load(train_weights, map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    model.eval()
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    xml_root = "./Pest24/Annotations"
    files = os.listdir(xml_root)
    files = files[:20]

    for file in files:
        if not file.endswith('xml'):
            continue
        xml_path = os.path.join(xml_root, file)
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = parse_xml_to_dict(xml)["annotation"]  # 读取的标注数据

        assert "object" in data, "{} lack of object information.".format(xml_path)
        name = data['filename']
        img_path = './Pest24/JPEGImages/{}.jpg'.format(name)
        # img = cv2.imread(img_path)
        original_img = cv2.imread(img_path)

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            predictions = model(img.to(device))[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
            # print(predict_scores)
            for i in range(len(predict_boxes)):
                if predict_scores[i] > 0.1:
                    box = predict_boxes[i]
                    cv2.rectangle(original_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            cv2.imwrite('./results/annotation/p1_{}.jpg'.format(name), original_img)


if __name__ == '__main__':
    read_predict()