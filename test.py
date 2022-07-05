import os
import time
import json
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import collections
import numpy as np

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box
from lxml import html
etree = html.etree

def parse_xml_to_dict(xml):

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


def xml_data(anno_path):
    # print(anno_path)
    annotations_root = "./Pest24/Annotations"  # XML文件路径
    file = anno_path[-11:-4]
    predictions = {}
    predict_boxes = []
    predict_classes = []
    predict_scores = []
    xml_path = os.path.join(annotations_root, "{}.xml".format(file))
    with open(xml_path, encoding='utf-8') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str.encode('utf-8'))
    data = parse_xml_to_dict(xml)["annotation"]  # 读取的标注数据
    for obj in data["object"]:
        predict_boxes.append([int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"]),
                              int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])])
        predict_classes.append(int(obj["name"]))
        predict_scores.append(1.00)

    predictions["boxes"] = predict_boxes
    predictions["labels"] = predict_classes
    predictions["scores"] = predict_scores

    # predictions.append({"boxes": predict_boxes},
    #                    {"labels": predict_classes},
    #                    {"scores": predict_scores})


    return predictions


def read_anno():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}
    # category_index = {k: v for k, v in class_dict.items()}

    ## 图片预测结果写入json文件
    img_root = "./Pest24/JPEGImages"  # 图片路径

    # read train.txt or val.txt file
    txt_path = "./Pest24/ImageSets/test.txt"  # train.txt文件路径，文件里是分割好的训练集图片名称
    assert os.path.exists(txt_path), "not found file."
    with open(txt_path) as read:
        img_list = [os.path.join(img_root, line.strip() + ".jpg")
                         for line in read.readlines() if len(line.strip()) > 0]  # img_list存储所有test集图片路径

    # check file
    assert len(img_list) > 0, "in '{}' file does not find any information.".format(txt_path)
    # load image
    thresh = 0.5
    img_list = img_list[40:60]
    for img_path in img_list:
        assert os.path.exists(img_path), "not found '{}' file.".format(img_path)
        original_img = Image.open(img_path)

        # from pil image to tensor, do not normalize image
        # data_transform = transforms.Compose([transforms.ToTensor()])
        # img = data_transform(original_img)
        # # expand batch dimension
        # img = torch.unsqueeze(img, dim=0)


        with torch.no_grad():
            # init
            # img_height, img_width = img.shape[-2:]
            # init_img = torch.zeros((1, 3, img_height, img_width), device=device)


            predictions = xml_data(img_path)


            predict_boxes = np.array(predictions["boxes"])
            predict_classes = np.array(predictions["labels"])
            predict_scores = np.array(predictions["scores"])
            #
            if len(predict_boxes) == 0:
                 print("没有检测到任何目标!")

            print(img_path, predict_boxes, predict_classes, predict_scores)

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=3)
            plt.imshow(original_img)
            plt.show()
            # 保存预测的图片结果
            original_img.save("./results1.0/gt/{}".format(img_path[-11:]))



def create_model(num_classes):

    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def predict_iou():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=25) # 24类+背景

    # load train weights
    train_weights = "./save_weights_iou/resNetFpn-model-20.pth"
    #train_weights = "./save_weights/"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    state_dict = torch.load(train_weights, map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    model.eval()

    # model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    # model.load_state_dict(torch.load(train_weights, map_location=device)["model"],strict=False)

    model.to(device)


    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}
    # category_index = {k: v for k, v in class_dict.items()}


    ## 图片预测结果写入json文件
    img_root = "./Pest24/JPEGImages"  # 图片路径
    # annotations_root = "./Pest24/Annotations"  # XML文件路径

    # read train.txt or val.txt file
    txt_path = "./Pest24/ImageSets/test.txt"  # train.txt文件路径，文件里是分割好的训练集图片名称
    assert os.path.exists(txt_path), "not found file."
    with open(txt_path) as read:
        img_list = [os.path.join(img_root, line.strip() + ".jpg")
                         for line in read.readlines() if len(line.strip()) > 0]  # img_list存储所有test集图片路径

    # check file
    assert len(img_list) > 0, "in '{}' file does not find any information.".format(txt_path)
    # load image
    thresh = 0.5
    img_list = img_list[:20]
    for img_path in img_list:
        assert os.path.exists(img_path), "not found '{}' file.".format(img_path)
        original_img = Image.open(img_path)

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

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            # print("inference+NMS time: {}".format(t_end - t_start))
            # print("{}\n{}\n".format(img_path,predictions))
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            #
            if len(predict_boxes) == 0:
                 print("没有检测到任何目标!")

            print(img_path, predict_boxes, predict_classes, predict_scores)

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=2)
            plt.imshow(original_img)
            plt.show()
            # 保存预测的图片结果
            original_img.save("./results1.0/predict_iou/{}".format(img_path[-11:]))

def main():
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

    # model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    # model.load_state_dict(torch.load(train_weights, map_location=device)["model"],strict=False)

    model.to(device)


    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}
    # category_index = {k: v for k, v in class_dict.items()}


    ## 图片预测结果写入json文件
    img_root = "./Pest24/JPEGImages"  # 图片路径
    # annotations_root = "./Pest24/Annotations"  # XML文件路径

    # read train.txt or val.txt file
    txt_path = "./Pest24/ImageSets/test.txt"  # train.txt文件路径，文件里是分割好的训练集图片名称
    assert os.path.exists(txt_path), "not found file."
    with open(txt_path) as read:
        img_list = [os.path.join(img_root, line.strip() + ".jpg")
                         for line in read.readlines() if len(line.strip()) > 0]  # img_list存储所有test集图片路径

    # check file
    assert len(img_list) > 0, "in '{}' file does not find any information.".format(txt_path)
    # load image
    thresh = 0.5
    img_list = img_list[:20]
    for img_path in img_list:
        assert os.path.exists(img_path), "not found '{}' file.".format(img_path)
        original_img = Image.open(img_path)

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

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            # print("inference+NMS time: {}".format(t_end - t_start))
            # print("{}\n{}\n".format(img_path,predictions))
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            #
            if len(predict_boxes) == 0:
                 print("没有检测到任何目标!")

            print(img_path, predict_boxes, predict_classes, predict_scores)

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=2)
            plt.imshow(original_img)
            plt.show()
            # 保存预测的图片结果
            original_img.save("./results1.0/predict_ours/{}".format(img_path[-11:]))



if __name__ == '__main__':
    # main()
    # predict_iou()
    read_anno()
