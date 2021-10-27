import os
import time
import json
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import collections

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=38) # 24类+背景

    # load train weights
    train_weights = "./save_weights/resNetFpn-model-14.pth"
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
    annotations_root = "./Pest24/Annotations"  # XML文件路径

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

            box_to_display_str_map = collections.defaultdict(list)
            box_to_color_map = collections.defaultdict(str)
            print(img_path[20:])
            for i in range(predict_boxes.shape[0]):
                box = tuple(predict_boxes[i].tolist())  # numpy -> list -> tuple
                if predict_scores[i] > thresh:
                    if predict_classes[i] in category_index.keys():
                        class_name = category_index[predict_classes[i]]
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
                    display_str = '{}: {}%'.format(display_str, int(100 * predict_scores[i]))
                    # box_to_display_str_map[box].append(display_str)
                    # box_to_color_map[box] = STANDARD_COLORS[
                    #     classes[i] % len(STANDARD_COLORS)]
                    print("{}\nbox:{}\n".format(display_str,box))
                else:
                    break  # 网络输出概率已经排序过，当遇到一个不满足后面的肯定不满足


            # print(img_path,predict_boxes,predict_classes,predict_scores)
            #
            # draw_box(original_img,
            #          predict_boxes,
            #          predict_classes,
            #          predict_scores,
            #          category_index,
            #          thresh=0.5,
            #          line_thickness=3)
            # plt.imshow(original_img)
            # plt.show()
            # # 保存预测的图片结果
            # original_img.save("test_result.jpg")




if __name__ == '__main__':
    main()

