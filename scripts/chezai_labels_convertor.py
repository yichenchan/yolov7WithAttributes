# -*- coding: utf-8 -*-
import sys 
import os
from xml.etree import ElementTree as etree
from xml.etree.ElementTree import Element,SubElement,ElementTree


#boxes_names=["car","car_face","car_butt","person","plate","left_signal","right_signal","seperate_area","road_block","traffic_light","highway_sign","city_sign"]
#23 类
boxes_names=["car_normal","car_seperator","car_zebra","car_body","car_face","car_butt","person","plate","zebra_crossing","left_signal","right_signal","seperate_area","grid_line","road_block","bus_lane","traffic_light","guide_line","motorcycle","motor_person","head","helmet","highway_sign","city_sign"]

xml_num = 1
image_num = 1

for folder, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        if "box.xml" in file:
            xml_path = os.path.join(os.path.abspath(folder), file)
            print(str(xml_num) + ": deal with " + xml_path)
            xml_num += 1

            #构建 xml 树
            with open(xml_path, 'r', encoding='utf-8') as xml_file:
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
                    for box in boxes:
                        # 获取当前box的坐标        
                        xtl = float(box.get('xtl'))
                        ytl = float(box.get('ytl'))
                        xbr = float(box.get('xbr')) 
                        ybr = float(box.get('ybr'))

                        #获取当前box的标签并进行标签合并处理
                        label = box.get('label')
                        # if label in ['car_normal', 'car_seperator', 'car_zebra', 'car_body']:
                        #     print("---> " + label + " to " + "car")
                        #     label = 'car'
                        if label in ['car_face', 'solid_car_face', 'dotted_car_face']:
                            print("---> " + label + " to " + "car_face")
                            label = 'car_face'
                        if label in ['car_butt', 'solid_car_butt', 'dotted_car_butt']:
                            print("---> " + label + " to " + "car_butt")
                            label = 'car_butt'
                        if label in ['left_person', 'right_person', 'normal_person']:
                            print("---> " + label + " to " + "person")
                            label = 'person'
                        if label in ['signal']:
                            print("---> " + label + " to " + "left_signal")
                            label = 'left_signal'
                        if label in ['traffic_light', 'p_traffic_light']:
                            print("---> " + label + " to " + "traffic_light")
                            label = "traffic_light"
                        
                        #构建 yolov5 支持的标签格式
                        objects.append([label, xtl, ytl, xbr, ybr])

                    gstr = ''
                    for obj in objects:
                        # 过滤不需要的标签
                        label, x1, y1, x2, y2 = obj
                        if not label in boxes_names:
                            continue
                        
                        # 对于标签坐标不合法的给予修正
                        x1 = x1 if x1 >= 0 else 0
                        y1 = y1 if y1 >= 0 else 0
                        x2 = x2 if x2 < width else width -1
                        y2 = y2 if y2 < height else height -1
                        x2 = x2 if x2 >= 0 else 0
                        y2 = y2 if y2 >= 0 else 0
                        x1 = x1 if x1 < width else width -1
                        y1 = y1 if y1 < height else height -1
                        print("---> " + str(boxes_names.index(label)) + ' ' + str((x1+x2)/2./width) + ' ' + str((y1+y2)/2./height) + ' ' + str(abs(x1-x2)/width) + ' ' + str(abs(y1-y2)/height))
                        gstr = gstr + str(boxes_names.index(label)) + ' ' + str((x1+x2)/2./width) + ' ' + str((y1+y2)/2./height) + ' ' + str(abs(x1-x2)/width) + ' ' + str(abs(y1-y2)/height) + '\n'

                    # 写入当前标签进 txt 文件
                    if os.path.exists(img_name):
                        # 如果存在之前的，先删除之前的
                        if os.path.exists(img_name[:-3] + 'txt'):
                            os.remove(img_name[:-3] + 'txt')
                            print("--->removed label " + img_name)
                        
                        # 写入新的标签
                        with open(img_name[:-3]+'txt','a') as f:
                            f.write(gstr)
                            print("--->written label " + img_name)

                    # 把当前文件写入 all.txt
                    with open('chezai_dataset.txt', 'a') as f:
                        f.write(img_name[:-3] + 'txt' + '\n')    
    
