import os
import xml.etree.ElementTree as ET
import re
import string
import numpy as np

#from .util import read_image
import torch


def check():
    xmlfile_dict = './Pest24/Annotations'

    label = list()
    objs = []
    names = []
    i = 0
    for file in os.listdir(xmlfile_dict):
        if file.endswith('xml'):
            anno = ET.parse(os.path.join(xmlfile_dict, file))
            bbox = list()

            difficult = list()
            for obj in anno.findall('object'):
                # when in not using difficult split, and the object is
                # difficult, skipt it.
                if int(obj.find('difficult').text) == 1:
                    continue
                # name = obj.find('name').text.lower().strip()  # 种类
                name = obj.find('name').text  # 种类
                name = str(name)
                # objs.append(name)
                i = 1
                for new in names:
                    if new == name:
                        i = 0
                if i == 1:
                    names.append(name)
                    # print(name)
                bbox = obj.find('bndbox')
                xmin, ymin, xmax, ymax = float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(
                    bbox.find('xmax').text), float(bbox.find('ymax').text)
                if xmin >= xmax:
                    print("{}{}xmin:{}>=xmax:{}".format(file, obj, xmin, xmax))
                if ymin >= ymax:
                    print("{}{}ymin:{}>=ymax:{}".format(file, obj, ymin, ymax))
                if xmin <= 0 or ymin <= 0 or xmax <= 0 or ymax <= 0:
                    print("{}{}xmin:{} xmax:{} ymin:{} ymax:{}".format(file, obj, xmin, xmax, ymin, ymax))

    for i in range(len(names)):
        print('''"{}": {},'''.format(names[i], i))


def readfile():
    file_name = open('./1w_rep1.txt')
    write_txt = './1w_rep2.txt'
    with open(write_txt, 'a+') as f:
        for line in file_name.readlines():
            line = line.strip('\n')
            strs = re.findall('\d+SM', line)
            if len(strs) > 0:
                str = strs[0]
                x = re.findall('\d+', str)
                num = int(x[0])
                if num >= 2:
                    f.write(line + '\n')
                    print(line)



if __name__ == '__main__':

    readfile()




