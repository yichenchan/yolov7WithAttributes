# -*- coding: utf-8 -*-
import sys 
import os
from xml.etree import ElementTree as etree
from xml.etree.ElementTree import Element,SubElement,ElementTree
import shutil


#boxes_names=["car","car_face","car_butt","person","plate","left_signal","right_signal","seperate_area","road_block","traffic_light","highway_sign","city_sign"]
#23 类
boxes_names=["car_normal","car_seperator","car_zebra","car_body","car_face","car_butt","person","plate","zebra_crossing","left_signal","right_signal","seperate_area","grid_line","road_block","bus_lane","traffic_light","guide_line","motorcycle","motor_person","head","helmet","highway_sign","city_sign"]
boxes_names_dict = {}

xml_num = 1
image_num = 1

for folder, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        if "box.xml" in file:
            xml_path = os.path.join(os.path.abspath(folder), file)
            print(str(xml_num) + ": deal with " + xml_path)
            xml_num += 1

            #构建 xml 树
            with open(xml_path, 'r') as xml_file:
                tree  = etree.parse(xml_file)
                root = tree.getroot()
                images = root.findall('image')

                for image in images:
                    # 获取各个元素
                    img_name = image.get('name')
                    width = int(image.get('width'))
                    height = int(image.get('height'))
                    boxes = image.findall('box')
                    lines = image.findall('polyline')

                    # 对于路径第一个文件夹不是 img 的给予修正
                    if img_name.split('/')[0] == 'img':
                        img_name = os.path.join(os.path.abspath(folder)) + '/' + img_name
                    else:
                        img_name = os.path.join(os.path.abspath(folder)) + '/img/' + img_name
                    print(str(image_num) + ": deal with " + img_name)
                    image_num += 1
                    
                    objects= []
                    needed_exit = False
                    for box in boxes:
                        # 获取当前box的坐标        
                        xtl = float(box.get('xtl'))
                        ytl = float(box.get('ytl'))
                        xbr = float(box.get('xbr')) 
                        ybr = float(box.get('ybr'))

                        big_enough = abs(ybr - xbr) * abs(ytl - xtl) > 10000

                        #获取当前box的标签并进行标签合并处理
                        label = box.get('label')

                        # if label not in boxes_names_dict.keys():
                        #     boxes_names_dict[label] = 0
                        # else:
                        #     boxes_names_dict[label] += 1

                        if (label in ['grid_line', 'bus_lane', 'seperate_area', 'left_person', 'right_person', 'highway_sign', 'cguide_line', 'dotted_car_face', 'car_seperator', 'guide_line', 'zebra_crossing', 'city_sign']):
                            print('needed label found!')
                            needed_exit = True

                    if needed_exit:
                        dir_path = '/workspace/data/selected_dataset/' + os.path.dirname(img_name)[31:] + '/'
                        if(not os.path.exists(dir_path)):
                            os.makedirs(dir_path)
                        shutil.copy(img_name, dir_path)
                        # 把当前文件写入 all.txt
                        with open('new_chezai_dataset.txt', 'a') as f:
                            f.write(img_name[:-3] + 'txt' + '\n') 
                        
print(boxes_names_dict)                            
        
