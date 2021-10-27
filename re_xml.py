import os
import xml.etree.ElementTree as ET

def main():
    xmlfile_dict = './Pest24/Annotations'

    category_dict = {'25': '18',
                     '28': '19',
                     '29': '20',
                     '31': '21',
                     '32': '22',
                     '34': '23',
                     '35': '17',
                     '36': '9',
                     '37': '4'}

    for file in os.listdir(xmlfile_dict):
        # print(file)
        if file.endswith('xml'):
            root = ET.parse(os.path.join(xmlfile_dict, file))
            for obj in root.findall('object'):
                # when in not using difficult split, and the object is
                # difficult, skipt it.
                category = obj.find('name')
                str = category.text
                if str in category_dict.keys():
                    # print(category, category_dict[category])
                    category.text = category_dict[str]
                    # obj.set('name', category_dict[category])

            # tree = ET.ElementTree(root)
            # ET.dump(root)
            with open(os.path.join(xmlfile_dict, file), 'wb') as f:
                root.write(f)

    print('write success')


if __name__ == '__main__':
    main()
