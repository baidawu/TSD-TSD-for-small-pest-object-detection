import os
import xml.etree.ElementTree as ET

import numpy as np

#from .util import read_image

if __name__ == '__main__':

    xmlfile_dict = './Pest24/Annotations'

    label = list()
    objs = []
    names = []
    i = 0
    for file in os.listdir(xmlfile_dict):
        if file.endswith('xml'):
            anno = ET.parse(os.path.join(xmlfile_dict,file))
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
                i=1
                for new in names:
                    if new == name:
                        i=0
                if i==1:
                    names.append(name)
                    # print(name)
                bbox = obj.find('bndbox')
                xmin, ymin, xmax, ymax = float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(bbox.find('xmax').text), float(bbox.find('ymax').text)
                if xmin >= xmax:
                    print("{}{}xmin:{}>=xmax:{}".format(file, obj, xmin, xmax))
                if ymin >= ymax:
                    print("{}{}ymin:{}>=ymax:{}".format(file, obj, ymin, ymax))
                if xmin <= 0 or ymin <= 0 or xmax <= 0 or ymax <= 0:
                    print("{}{}xmin:{} xmax:{} ymin:{} ymax:{}".format(file, obj, xmin, xmax, ymin, ymax))


    for i in range(len(names)):
        print('''"{}": {},'''.format(names[i],i))

    # print(names)
