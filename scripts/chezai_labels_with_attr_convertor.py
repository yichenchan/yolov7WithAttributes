# -*- coding: utf-8 -*-
import sys 
import os
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element,SubElement,ElementTree
import json

major_classes_names = [
    "vehicle",
    "person",
    "bus_station",
    "traffic_light",
    "guide_line",
    "road_sign",
    "bus_lane",
    "zebra_line",
    "grid_line",
    "diversion_line"
]

car_type = ["car", "bus", "truck"]
person_type = ["person", "traffic_police"]
light_state = ["no_bright", "bright"]
helmet_state = ["no_wear", "wear"]
person_state = ["person", "motor_person"]
traffic_light_color = ["green", "red", "yellow"]
road_sign_color = ["blue", "green"]


if os.path.exists('chezai_dataset_with_attr.txt'):
    os.remove('chezai_dataset_with_attr.txt')
    print("--->removed all_dataset.txt " + 'chezai_dataset_with_attr.txt')


dataset_dir = sys.argv[1]

for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    print("deal with " + folder_path)

    if os.path.isdir(folder_path):
        annotation_file = os.path.join(folder_path, 'annotations.xml')
        if not os.path.exists(annotation_file):
            continue

        tree = ET.parse(annotation_file)
        root = tree.getroot()

        for image in root.findall('image'):
            print("deal with " + os.path.join(folder_path, image.get('name')))
            width = int(image.get('width'))
            height = int(image.get('height'))

            boxes = image.findall('box')

            major_classes = []
            minor_classes = {}
            for box in boxes:
                if(box.get('label') in major_classes_names):
                    major_classes.append(box)
                else:
                    attr_name = str(box.get('label'))
                    attr_id = int(box.find('attribute[@name="id"]').text)
                    attr_box_x1 = float(box.get('xtl'))
                    attr_box_y1 = float(box.get('ytl'))
                    attr_box_x2 = float(box.get('xbr'))
                    attr_box_y2 = float(box.get('ybr'))
                    attr_box_x1 = attr_box_x1 if attr_box_x1 >= 0 else 0
                    attr_box_y1 = attr_box_y1 if attr_box_y1 >= 0 else 0
                    attr_box_x2 = attr_box_x2 if attr_box_x2 < width else width -1
                    attr_box_y2 = attr_box_y2 if attr_box_y2 < height else height -1
                    attr_box_x_normalized = float(attr_box_x1 / width)
                    attr_box_y_normalized = float(attr_box_y1 / height)
                    attr_box_w_normalized = float((attr_box_x2 - attr_box_x1) / width)
                    attr_box_h_normalized = float((attr_box_y2 - attr_box_y1) / height)
                    minor_classes[attr_name + '_' + str(attr_id)] = (attr_box_x_normalized, attr_box_y_normalized, attr_box_w_normalized, attr_box_h_normalized)

            curr_img_labels = []
            for box in major_classes:
                # ------index 0: 类别
                class_index_0 = str(major_classes_names.index(box.get('label')))
                print(box.get('label'), class_index_0)

                # ------index 1~4: 检测框坐标
                x1 = float(box.get('xtl'))
                y1 = float(box.get('ytl'))
                x2 = float(box.get('xbr'))
                y2 = float(box.get('ybr'))
                x1 = x1 if x1 >= 0 else 0
                y1 = y1 if y1 >= 0 else 0
                x2 = x2 if x2 < width else width -1
                y2 = y2 if y2 < height else height -1
                x_normalized_1 = float(x1 / width)
                y_normalized_2 = float(y1 / height)
                w_normalized_3 = float((x2 - x1) / width)
                h_normalized_4 = float((y2 - y1) / height)

                # ------index 5~32: 车辆相关属性
                # 车辆类别（5），
                # 左车灯是否可见（6），左车灯状态（7），左车灯位置（8~11），
                # 右车灯是否可见（12），右车灯状态（13），右车灯位置（14~17），
                # 车屁股是否可见（18），车屁股框位置（19~22）
                # 车头是否可见（23），车头框位置（24~27）
                # 车牌是否可见（28），车牌位置（29~32），
                vehicle_attr_5_to_32 = [-1] * 28
                if(box.get('label') == 'vehicle'):
                    box_id = -1
                    attributes = box.findall('attribute')
                    # 进行属性可见性判断、获取id
                    for attribute in attributes:
                        if(attribute.get('name') == "car_type"):
                            vehicle_attr_5_to_32[5 - 5] = int(car_type.index(attribute.text))
                        if(attribute.get('name') == "left_light"):
                            vehicle_attr_5_to_32[7 - 5] = int(light_state.index(attribute.text))
                        if(attribute.get('name') == "right_light"):
                            vehicle_attr_5_to_32[13 - 5] = int(light_state.index(attribute.text))
                        if(attribute.get('name') == 'id'):
                            box_id = int(attribute.text)
                    # 如果左车灯可见，去 minor_classes 中寻找对应属性 box
                    if("left_light_" + str(box_id) in minor_classes.keys()):
                        left_light_box = minor_classes["left_light_" + str(box_id)]
                        vehicle_attr_5_to_32[6 - 5] = 1
                        vehicle_attr_5_to_32[8 - 5] = left_light_box[0] - x_normalized_1 if left_light_box[0] - x_normalized_1 > 0 else 0
                        vehicle_attr_5_to_32[9 - 5] = left_light_box[1] - y_normalized_2 if left_light_box[1] - y_normalized_2 > 0 else 0
                        vehicle_attr_5_to_32[10 - 5] = left_light_box[2]
                        vehicle_attr_5_to_32[11 - 5] = left_light_box[3]
                    else:
                        vehicle_attr_5_to_32[6 - 5] = 0
                    # 如果右车灯可见，去 minor_classes 中寻找对应属性 box
                    if("right_light_" + str(box_id) in minor_classes.keys()):
                        right_light_box = minor_classes["right_light_" + str(box_id)]
                        vehicle_attr_5_to_32[12 - 5] = 1
                        vehicle_attr_5_to_32[14 - 5] = right_light_box[0] - x_normalized_1 if right_light_box[0] - x_normalized_1 > 0 else 0
                        vehicle_attr_5_to_32[15 - 5] = right_light_box[1] - y_normalized_2 if right_light_box[1] - y_normalized_2 > 0 else 0
                        vehicle_attr_5_to_32[16 - 5] = right_light_box[2]
                        vehicle_attr_5_to_32[17 - 5] = right_light_box[3]
                    else:
                        vehicle_attr_5_to_32[12 - 5] = 0
                    
                    # 如果车屁股可见，去 minor_classes 中寻找对应属性 box
                    if("vehicle_butt_" + str(box_id) in minor_classes.keys()):
                        vehicle_butt_box = minor_classes["vehicle_butt_" + str(box_id)]
                        vehicle_attr_5_to_32[18 - 5] = 1
                        vehicle_attr_5_to_32[19 - 5] = vehicle_butt_box[0] - x_normalized_1 if vehicle_butt_box[0] - x_normalized_1 > 0 else 0
                        vehicle_attr_5_to_32[20 - 5] = vehicle_butt_box[1] - y_normalized_2 if vehicle_butt_box[1] - y_normalized_2 > 0 else 0
                        vehicle_attr_5_to_32[21 - 5] = vehicle_butt_box[2]
                        vehicle_attr_5_to_32[22 - 5] = vehicle_butt_box[3]
                    else:
                        vehicle_attr_5_to_32[18 - 5] = 0

                    # 如果车头可见，去 minor_classes 中寻找对应属性 box
                    if("vehicle_head_" + str(box_id) in minor_classes.keys()):
                        vehicle_head_box = minor_classes["vehicle_head_" + str(box_id)]
                        vehicle_attr_5_to_32[23 - 5] = 1
                        vehicle_attr_5_to_32[24 - 5] = vehicle_head_box[0] - x_normalized_1 if vehicle_head_box[0] - x_normalized_1 > 0 else 0
                        vehicle_attr_5_to_32[25 - 5] = vehicle_head_box[1] - y_normalized_2 if vehicle_head_box[1] - y_normalized_2 > 0 else 0
                        vehicle_attr_5_to_32[26 - 5] = vehicle_head_box[2]
                        vehicle_attr_5_to_32[27 - 5] = vehicle_head_box[3]
                    else:
                        vehicle_attr_5_to_32[23 - 5] = 0

                    # 如果车牌可见，去 minor_classes 中寻找对应属性 box
                    if("plate_" + str(box_id) in minor_classes.keys()):
                        plate_box = minor_classes["plate_" + str(box_id)]
                        vehicle_attr_5_to_32[28 - 5] = 1
                        vehicle_attr_5_to_32[29 - 5] = plate_box[0] - x_normalized_1 if plate_box[0] - x_normalized_1 > 0 else 0
                        vehicle_attr_5_to_32[30 - 5] = plate_box[1] - y_normalized_2 if plate_box[1] - y_normalized_2 > 0 else 0
                        vehicle_attr_5_to_32[31 - 5] = plate_box[2]
                        vehicle_attr_5_to_32[32 - 5] = plate_box[3]
                    else:
                        vehicle_attr_5_to_32[28 - 5] = 0

                # ------ index 33~50: 行人相关属性
                # 人的类别（33），人的状态（34）， 是否带头盔（35）
                # 人头是否可见（36），人头的位置（37~40），
                # 人骑的车是否可见（41），人骑的车的位置（42~45）
                # 电动车车牌是否可见（46），车牌的位置（47~50），
                person_attr_33_to_50 = [-1] * 18
                if(box.get('label') == 'person'):
                    box_id = -1
                    attributes = box.findall('attribute')
                    # 进行属性可见性判断、获取id
                    for attribute in attributes:
                        if(attribute.get('name') == "person_type"):
                            person_attr_33_to_50[33 - 33] = int(person_type.index(attribute.text))
                        if(attribute.get('name') == "state"):
                            if attribute.text in road_sign_color:
                                person_attr_33_to_50[34 - 33] = int(person_state.index(attribute.text))
                            else:
                                person_attr_33_to_50[34 - 33] = 0
                        if(attribute.get('name') == "helmet"):
                            person_attr_33_to_50[35 - 33] = int(helmet_state.index(attribute.text))
                        if(attribute.get('name') == 'id'):
                            box_id = int(attribute.text)
                    # 如果人头可见，去 minor_classes 中寻找对应属性 box
                    if("person_head_" + str(box_id) in minor_classes.keys()):
                        person_head_box = minor_classes["person_head_" + str(box_id)]
                        person_attr_33_to_50[36 - 33] = 1
                        person_attr_33_to_50[37 - 33] = person_head_box[0] - x_normalized_1 if person_head_box[0] - x_normalized_1 > 0 else 0
                        person_attr_33_to_50[38 - 33] = person_head_box[1] - y_normalized_2 if person_head_box[1] - y_normalized_2 > 0 else 0
                        person_attr_33_to_50[39 - 33] = person_head_box[2]
                        person_attr_33_to_50[40 - 33] = person_head_box[3]
                    else:
                        person_attr_33_to_50[36 - 33] = 0
                    # 如果人骑的车可见，去 minor_classes 中寻找对应属性 box
                    if("non_motor_" + str(box_id) in minor_classes.keys()):
                        non_motor_box = minor_classes["non_motor_" + str(box_id)]
                        person_attr_33_to_50[41 - 33] = 1
                        person_attr_33_to_50[42 - 33] = non_motor_box[0] - x_normalized_1 if non_motor_box[0] - x_normalized_1 > 0 else 0
                        person_attr_33_to_50[43 - 33] = non_motor_box[1] - y_normalized_2 if non_motor_box[1] - y_normalized_2 > 0 else 0
                        person_attr_33_to_50[44 - 33] = non_motor_box[2]
                        person_attr_33_to_50[45 - 33] = non_motor_box[3]
                    else:
                        person_attr_33_to_50[41 - 33] = 0
                    # 如果人骑的车的车牌可见，去 minor_classes 中寻找对应属性 box
                    if("non_motor_plate_" + str(box_id) in minor_classes.keys()):
                        non_motor_plate_box = minor_classes["non_motor_plate_" + str(box_id)]
                        person_attr_33_to_50[46 - 33] = 1
                        person_attr_33_to_50[47 - 33] = non_motor_plate_box[0] - x_normalized_1 if non_motor_plate_box[0] - x_normalized_1 > 0 else 0
                        person_attr_33_to_50[48 - 33] = non_motor_plate_box[1] - y_normalized_2 if non_motor_plate_box[1] - y_normalized_2 > 0 else 0
                        person_attr_33_to_50[49 - 33] = non_motor_plate_box[2]
                        person_attr_33_to_50[50 - 33] = non_motor_plate_box[3]
                    else:
                        person_attr_33_to_50[46 - 33] = 0


                # ------ index 51: 红绿灯的颜色
                traffic_light_attr_51 = -1
                if(box.get('label') == 'traffic_light'):
                    if attribute.text  in road_sign_color:
                        traffic_light_attr = int(traffic_light_color.index(attribute.text))
                    else:
                        traffic_light_attr = 0

                # ------ index 52: 路牌颜色
                road_sign_attr_52 = -1
                if(box.get('label') == 'road_sign'):
                    if attribute.text  in road_sign_color:
                        road_sign_attr = int(road_sign_color.index(attribute.text))
                    else:
                        road_sign_attr = 0

                # ------ index 53~60: 公交车道4个点坐标
                # ------ index 61~68: 斑马线4个点坐标
                # ------ index 69~76: 网格线4个点坐标
                # ------ index 77~84: 导流线4个点坐标
                # 由于 xml 文件中不含最后四类的属性，所以全部置为 -1
                lanes_attr_53_to_84 = [-1] * 32

                curr_box_labels = [class_index_0, x_normalized_1, y_normalized_2, w_normalized_3, h_normalized_4] + \
                                    vehicle_attr_5_to_32 + \
                                    person_attr_33_to_50 + \
                                    [traffic_light_attr_51] + \
                                    [road_sign_attr_52] + \
                                    lanes_attr_53_to_84
                
                # 确保标签个数为 85
                assert len(curr_box_labels) == 85

                curr_img_labels.append(curr_box_labels)

            # # 读取 json 最后四类检测对象的数据
            # json_path = xml_path[:-3] + 'json'
            # with open(json_path, 'r', encoding='utf-8') as json_file:
            #     root = json.load(json_file)
            #     annotations = root['annotations']
            #     for box in annotations:
            #         # ------index 0: 类别
            #         class_index_0 = str(major_classes_names.index(box['category_name']))
            #         print(box['category_name'], class_index_0)

            #         # ------index 1~4: 检测框坐标
            #         x1 = float(box['bbox'][0])
            #         y1 = float(box['bbox'][1])
            #         x2 = float(box['bbox'][2])
            #         y2 = float(box['bbox'][3])
            #         x1 = x1 if x1 >= 0 else 0
            #         y1 = y1 if y1 >= 0 else 0
            #         x2 = x2 if x2 < width else width -1
            #         y2 = y2 if y2 < height else height -1
            #         x_normalized_1 = float(x1 / width)
            #         y_normalized_2 = float(y1 / height)
            #         w_normalized_3 = float((x2 - x1) / width)
            #         h_normalized_4 = float((y2 - y1) / height)

            #         # -----index 5~52: 不可能出现这些类别，所以全部置 -1
            #         attr_before_lane = [-1] * 48

            #         # ------ index 53~60: 公交车道4个点坐标
            #         bus_lane_attr = [-1] * 8
            #         if(box['category_name'] == 'bus_lane'):
            #             bus_lane_attr[0] = max((box['keypoints'][0] / width) - x_normalized_1, 0)
            #             bus_lane_attr[1] = max((box['keypoints'][1] / height) - y_normalized_2, 0)
            #             bus_lane_attr[2] = min((box['keypoints'][2] / width) - x_normalized_1, 1)
            #             bus_lane_attr[3] = max((box['keypoints'][3] / height) - y_normalized_2, 0)
            #             bus_lane_attr[4] = min((box['keypoints'][4] / width) - x_normalized_1, 1)
            #             bus_lane_attr[5] = min((box['keypoints'][5] / height) - y_normalized_2, 1)
            #             bus_lane_attr[6] = max((box['keypoints'][6] / width) - x_normalized_1, 0)
            #             bus_lane_attr[7] = min((box['keypoints'][7] / height) - y_normalized_2, 1)

            #         # ------ index 61~68: 斑马线4个点坐标
            #         zebra_line_attr = [-1] * 8
            #         if(box['category_name'] == 'zebra_line'):
            #             zebra_line_attr[0] = max((box['keypoints'][0] / width) - x_normalized_1, 0)
            #             zebra_line_attr[1] = max((box['keypoints'][1] / height) - y_normalized_2, 0)
            #             zebra_line_attr[2] = min((box['keypoints'][2] / width) - x_normalized_1, 1)
            #             zebra_line_attr[3] = max((box['keypoints'][3] / height) - y_normalized_2, 0)
            #             zebra_line_attr[4] = min((box['keypoints'][4] / width) - x_normalized_1, 1)
            #             zebra_line_attr[5] = min((box['keypoints'][5] / height) - y_normalized_2, 1)
            #             zebra_line_attr[6] = max((box['keypoints'][6] / width) - x_normalized_1, 0)
            #             zebra_line_attr[7] = min((box['keypoints'][7] / height) - y_normalized_2, 1)

            #         # ------ index 69~76: 网格线4个点坐标
            #         grid_line_attr = [-1] * 8
            #         if(box['category_name'] == 'grid_line'):
            #             grid_line_attr[0] = max((box['keypoints'][0] / width) - x_normalized_1, 0)
            #             grid_line_attr[1] = max((box['keypoints'][1] / height) - y_normalized_2, 0)
            #             grid_line_attr[2] = min((box['keypoints'][2] / width) - x_normalized_1, 1)
            #             grid_line_attr[3] = max((box['keypoints'][3] / height) - y_normalized_2, 0)
            #             grid_line_attr[4] = min((box['keypoints'][4] / width) - x_normalized_1, 1)
            #             grid_line_attr[5] = min((box['keypoints'][5] / height) - y_normalized_2, 1)
            #             grid_line_attr[6] = max((box['keypoints'][6] / width) - x_normalized_1, 0)
            #             grid_line_attr[7] = min((box['keypoints'][7] / height) - y_normalized_2, 1)

            #         # ------ index 77~84: 导流线4个点坐标
            #         diversion_line_attr = [-1] * 8
            #         if(box['category_name'] == 'diversion_line'):
            #             diversion_line_attr[0] = max((box['keypoints'][0] / width) - x_normalized_1, 0)
            #             diversion_line_attr[1] = max((box['keypoints'][1] / height) - y_normalized_2, 0)
            #             diversion_line_attr[2] = min((box['keypoints'][2] / width) - x_normalized_1, 1)
            #             diversion_line_attr[3] = max((box['keypoints'][3] / height) - y_normalized_2, 0)
            #             diversion_line_attr[4] = min((box['keypoints'][4] / width) - x_normalized_1, 1)
            #             diversion_line_attr[5] = min((box['keypoints'][5] / height) - y_normalized_2, 1)
            #             diversion_line_attr[6] = max((box['keypoints'][6] / width) - x_normalized_1, 0)
            #             diversion_line_attr[7] = min((box['keypoints'][7] / height) - y_normalized_2, 1)

            #         curr_box_labels = [class_index_0, x_normalized_1, y_normalized_2, w_normalized_3, h_normalized_4] + \
            #                             attr_before_lane + \
            #                             bus_lane_attr + \
            #                             zebra_line_attr + \
            #                             grid_line_attr + \
            #                             diversion_line_attr
        
            #         # 确保标签个数为
            #         assert len(curr_box_labels) == 85

            #         curr_img_labels.append(curr_box_labels)

            # 写入当前标签进 txt 文件
            img_path = os.path.join(folder_path, image.get('name'))
            txt_path = img_path[:-3] + 'txt'
            if os.path.exists(img_path):
                # 如果存在之前的，先删除之前的
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    print("--->removed label " + txt_path)
                
                # 写入新的标签
                if len(curr_img_labels) != 0:
                    with open(txt_path,'a') as curr_img_labels_file:
                        for box_labels in curr_img_labels:
                            curr_box_written_str = " ".join(str(x) for x in box_labels) + '\n' 
                            curr_img_labels_file.write(curr_box_written_str)
                        print("--->written label " + txt_path)

                    # 把当前文件写入 all.txt
                    with open('chezai_dataset_with_attr.txt', 'a') as all_path_files:
                        print(img_path + "   written in chezai_dataset_with_attr.txt")
                        all_path_files.write(img_path + '\n')    
                else:
                    print("no labels!!")
