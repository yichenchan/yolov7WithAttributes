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
            curr_image_process_done = True

            image_sub_path =  image.get('name').split('/', 1)[1]
            if('img/' not in image_sub_path):
                image_sub_path = "img/" + image_sub_path
            image_full_path = os.path.join(folder_path, image_sub_path)
            print("deal with " + image_full_path)

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
                    attr_box_x1 = max(float(box.get('xtl')), 0) 
                    attr_box_y1 = max(float(box.get('ytl')), 0) 
                    attr_box_x2 = min(float(box.get('xbr')), width -1)
                    attr_box_y2 = min(float(box.get('ybr')), height -1)
                    attr_box_tl_x = float(attr_box_x1)
                    attr_box_tl_y = float(attr_box_y1)
                    attr_box_w = float(attr_box_x2 - attr_box_x1)
                    attr_box_h = float(attr_box_y2 - attr_box_y1)
                    minor_classes[attr_name + '_' + str(attr_id)] = (attr_box_tl_x, attr_box_tl_y, attr_box_w, attr_box_h)

            curr_img_labels = []
            for box in major_classes:
                # ------index 0: 类别
                class_index_0 = str(major_classes_names.index(box.get('label')))
                print(box.get('label'), class_index_0)

                # ------index 1~4: 检测框坐标
                x1 = max(float(box.get('xtl')), 0) 
                y1 = max(float(box.get('ytl')), 0) 
                x2 = min(float(box.get('xbr')), width -1)
                y2 = min(float(box.get('ybr')), height -1)
                box_tl_x = float(x1)
                box_tl_y = float(y1)
                box_w = float(x2 - x1)
                box_h = float(y2 - y1)
                x_normalized_centre_1 = float(float((x1 + x2) / 2) / width)
                y_normalized_centre_2 = float(float((y1 + y2) / 2) / height)
                w_normalized_3 = float(box_w / width)
                h_normalized_4 = float(box_h / height)

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
                            if(attribute.text == "no-bright"):
                                attribute.text = "no_bright"
                            vehicle_attr_5_to_32[7 - 5] = int(light_state.index(attribute.text))
                        if(attribute.get('name') == "right_light"):
                            vehicle_attr_5_to_32[13 - 5] = int(light_state.index(attribute.text))
                        if(attribute.get('name') == 'id'):
                            box_id = int(attribute.text)
                    # 如果左车灯可见，去 minor_classes 中寻找对应属性 box
                    if("left_light_" + str(box_id) in minor_classes.keys()):
                        left_light_box = minor_classes["left_light_" + str(box_id)]
                        vehicle_attr_5_to_32[6 - 5] = 1
                        vehicle_attr_5_to_32[8 - 5] = float(min(box_w, max(left_light_box[0] - box_tl_x, 0)) / box_w)
                        vehicle_attr_5_to_32[9 - 5] = float(min(box_h, max(left_light_box[1] - box_tl_y, 0)) / box_h)
                        vehicle_attr_5_to_32[10 - 5] = float((min(left_light_box[2] + left_light_box[0], box_tl_x + box_w) - left_light_box[0]) / box_w) 
                        vehicle_attr_5_to_32[11 - 5] = float((min(left_light_box[3] + left_light_box[1], box_tl_y + box_h) - left_light_box[1]) / box_h)
                    else:
                        vehicle_attr_5_to_32[6 - 5] = 0
                    # 如果右车灯可见，去 minor_classes 中寻找对应属性 box
                    if("right_light_" + str(box_id) in minor_classes.keys()):
                        right_light_box = minor_classes["right_light_" + str(box_id)]
                        vehicle_attr_5_to_32[12 - 5] = 1
                        vehicle_attr_5_to_32[14 - 5] = float(min(box_w, max(right_light_box[0] - box_tl_x, 0)) / box_w)
                        vehicle_attr_5_to_32[15 - 5] = float(min(box_h, max(right_light_box[1] - box_tl_y, 0)) / box_h)
                        vehicle_attr_5_to_32[16 - 5] = float((min(right_light_box[2] + right_light_box[0], box_tl_x + box_w) - right_light_box[0]) / box_w)
                        vehicle_attr_5_to_32[17 - 5] = float((min(right_light_box[3] + right_light_box[1], box_tl_y + box_h) - right_light_box[1]) / box_h)
                    else:
                        vehicle_attr_5_to_32[12 - 5] = 0
                    
                    # 如果车屁股可见，去 minor_classes 中寻找对应属性 box
                    if("vehicle_butt_" + str(box_id) in minor_classes.keys()):
                        vehicle_butt_box = minor_classes["vehicle_butt_" + str(box_id)]
                        vehicle_attr_5_to_32[18 - 5] = 1
                        vehicle_attr_5_to_32[19 - 5] = float(min(box_w, max(vehicle_butt_box[0] - box_tl_x, 0)) / box_w)
                        vehicle_attr_5_to_32[20 - 5] = float(min(box_h, max(vehicle_butt_box[1] - box_tl_y, 0)) / box_h)
                        vehicle_attr_5_to_32[21 - 5] = float((min(vehicle_butt_box[2] + vehicle_butt_box[0], box_tl_x + box_w) - vehicle_butt_box[0]) / box_w)
                        vehicle_attr_5_to_32[22 - 5] = float((min(vehicle_butt_box[3] + vehicle_butt_box[1], box_tl_y + box_h) - vehicle_butt_box[1]) / box_h)
                    else:
                        vehicle_attr_5_to_32[18 - 5] = 0

                    # 如果车头可见，去 minor_classes 中寻找对应属性 box
                    if("vehicle_head_" + str(box_id) in minor_classes.keys()):
                        vehicle_head_box = minor_classes["vehicle_head_" + str(box_id)]
                        vehicle_attr_5_to_32[23 - 5] = 1
                        vehicle_attr_5_to_32[24 - 5] = float(min(box_w, max(vehicle_head_box[0] - box_tl_x, 0)) / box_w)
                        vehicle_attr_5_to_32[25 - 5] = float(min(box_h, max(vehicle_head_box[1] - box_tl_y, 0)) / box_h)
                        vehicle_attr_5_to_32[26 - 5] = float((min(vehicle_head_box[2] + vehicle_head_box[0], box_tl_x + box_w) - vehicle_head_box[0]) / box_w)
                        vehicle_attr_5_to_32[27 - 5] = float((min(vehicle_head_box[3] + vehicle_head_box[1], box_tl_y + box_h) - vehicle_head_box[1]) / box_h)
                    else:
                        vehicle_attr_5_to_32[23 - 5] = 0

                    # 如果车牌可见，去 minor_classes 中寻找对应属性 box
                    if("plate_" + str(box_id) in minor_classes.keys()):
                        plate_box = minor_classes["plate_" + str(box_id)]
                        vehicle_attr_5_to_32[28 - 5] = 1
                        vehicle_attr_5_to_32[29 - 5] = float(min(box_w, max(plate_box[0] - box_tl_x, 0)) / box_w)
                        vehicle_attr_5_to_32[30 - 5] = float(min(box_h, max(plate_box[1] - box_tl_y, 0)) / box_h)
                        vehicle_attr_5_to_32[31 - 5] = float((min(plate_box[2] + plate_box[0], box_tl_x + box_w) - plate_box[0]) / box_w)
                        vehicle_attr_5_to_32[32 - 5] = float((min(plate_box[3] + plate_box[1], box_tl_y + box_h) - plate_box[1]) / box_h)
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
                            if attribute.text in person_state:
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
                        person_attr_33_to_50[37 - 33] = float(min(box_w, max(person_head_box[0] - box_tl_x, 0)) / box_w)
                        person_attr_33_to_50[38 - 33] = float(min(box_h, max(person_head_box[1] - box_tl_y, 0)) / box_h)
                        person_attr_33_to_50[39 - 33] = float((min(person_head_box[2] + person_head_box[0], box_tl_x + box_w) - person_head_box[0]) / box_w)
                        person_attr_33_to_50[40 - 33] = float((min(person_head_box[3] + person_head_box[1], box_tl_y + box_h) - person_head_box[1]) / box_h)
                    else:
                        person_attr_33_to_50[36 - 33] = 0
                    # 如果人骑的车可见，去 minor_classes 中寻找对应属性 box
                    if("non_motor_" + str(box_id) in minor_classes.keys()):
                        non_motor_box = minor_classes["non_motor_" + str(box_id)]
                        person_attr_33_to_50[41 - 33] = 1
                        person_attr_33_to_50[42 - 33] = float(min(box_w, max(non_motor_box[0] - box_tl_x, 0)) / box_w)
                        person_attr_33_to_50[43 - 33] = float(min(box_h, max(non_motor_box[1] - box_tl_y, 0)) / box_h)
                        person_attr_33_to_50[44 - 33] = float((min(non_motor_box[2] + non_motor_box[0], box_tl_x + box_w) - non_motor_box[0]) / box_w)
                        person_attr_33_to_50[45 - 33] = float((min(non_motor_box[3] + non_motor_box[1], box_tl_y + box_h) - non_motor_box[1]) / box_h)
                    else:
                        person_attr_33_to_50[41 - 33] = 0
                    # 如果人骑的车的车牌可见，去 minor_classes 中寻找对应属性 box
                    if("non_motor_plate_" + str(box_id) in minor_classes.keys()):
                        non_motor_plate_box = minor_classes["non_motor_plate_" + str(box_id)]
                        person_attr_33_to_50[46 - 33] = 1
                        person_attr_33_to_50[47 - 33] = float(min(box_w, max(non_motor_plate_box[0] - box_tl_x, 0)) / box_w)
                        person_attr_33_to_50[48 - 33] = float(min(box_h, max(non_motor_plate_box[1] - box_tl_y, 0)) / box_h)
                        person_attr_33_to_50[49 - 33] = float((min(non_motor_plate_box[2] + non_motor_plate_box[0], box_tl_x + box_w) - non_motor_plate_box[0]) / box_w)
                        person_attr_33_to_50[50 - 33] = float((min(non_motor_plate_box[3] + non_motor_plate_box[1], box_tl_y + box_h) - non_motor_plate_box[1]) / box_h)
                    else:
                        person_attr_33_to_50[46 - 33] = 0

                # ------ index 51: 红绿灯的颜色
                traffic_light_attr_51 = -1
                if(box.get('label') == 'traffic_light'):
                    if box.find('attribute[@name="color"]').text in traffic_light_color:
                        traffic_light_attr_51 = int(traffic_light_color.index(box.find('attribute[@name="color"]').text))
                    else:
                        traffic_light_attr_51 = 0

                # ------ index 52: 路牌颜色
                road_sign_attr_52 = -1
                if(box.get('label') == 'road_sign'):
                    if box.find('attribute[@name="color"]').text in road_sign_color:
                        road_sign_attr_52 = int(road_sign_color.index(box.find('attribute[@name="color"]').text))
                    else:
                        road_sign_attr_52 = 0

                # ------ index 53~60: 公交车道4个点坐标
                # ------ index 61~68: 斑马线4个点坐标
                # ------ index 69~76: 网格线4个点坐标
                # ------ index 77~84: 导流线4个点坐标
                # 由于 xml 文件中不含最后四类的属性，所以全部置为 -1
                lanes_attr_53_to_84 = [-1] * 32

                curr_box_labels = [class_index_0, x_normalized_centre_1, y_normalized_centre_2, w_normalized_3, h_normalized_4] + \
                                    vehicle_attr_5_to_32 + \
                                    person_attr_33_to_50 + \
                                    [traffic_light_attr_51] + \
                                    [road_sign_attr_52] + \
                                    lanes_attr_53_to_84
                
                # 确保标签个数为 85
                assert len(curr_box_labels) == 85

                curr_img_labels.append(curr_box_labels)

            # 读取 json 最后四类检测对象的数据
            json_path = image_full_path[:-3] + 'json'
            if(not os.path.exists(json_path)):
                curr_image_process_done = False
                continue

            with open(json_path, 'r', encoding='utf-8') as json_file:
                root = json.load(json_file)
                annotations = root['annotations']
                for box in annotations:
                    if(len(box['keypoints']) == 0):
                        curr_image_process_done = False
                        continue

                    # ------index 0: 类别
                    class_index_0 = str(major_classes_names.index(box['category_name']))
                    print(box['category_name'], class_index_0)

                    # ------index 1~4: 检测框坐标
                    box_tl_x = max(float(box['bbox'][0]), 0) 
                    box_tl_y = max(float(box['bbox'][1]), 0) 
                    box_w = float(box['bbox'][2]) + 0.0000001
                    box_h = float(box['bbox'][3]) + 0.0000001
                    x_normalized_centre_1 = float((box_tl_x + (box_w / 2)) / width)
                    y_normalized_centre_2 = float((box_tl_y + (box_h / 2)) / height)
                    w_normalized_3 = float(box_w / width)
                    h_normalized_4 = float(box_h / height)

                    # -----index 5~52: 不可能出现这些类别，所以全部置 -1
                    attr_before_lane = [-1] * 48

                    # ------ index 53~60: 公交车道4个点坐标
                    bus_lane_attr = [-1] * 8
                    if(box['category_name'] == 'bus_lane'):
                        bus_lane_attr[0] = float(min(box_w, max(box['keypoints'][0] - box_tl_x, 0)) / box_w)
                        bus_lane_attr[1] = float(min(box_h, max(box['keypoints'][1] - box_tl_y, 0)) / box_h)
                        bus_lane_attr[2] = float(min(box_w, max(box['keypoints'][3] - box_tl_x, 0)) / box_w)
                        bus_lane_attr[3] = float(min(box_h, max(box['keypoints'][4] - box_tl_y, 0)) / box_h)
                        bus_lane_attr[4] = float(min(box_w, max(box['keypoints'][6] - box_tl_x, 0)) / box_w)
                        bus_lane_attr[5] = float(min(box_h, max(box['keypoints'][7] - box_tl_y, 0)) / box_h)
                        bus_lane_attr[6] = float(min(box_w, max(box['keypoints'][9] - box_tl_x, 0)) / box_w)
                        bus_lane_attr[7] = float(min(box_h, max(box['keypoints'][10] - box_tl_y, 0)) / box_h)

                    # ------ index 61~68: 斑马线4个点坐标
                    zebra_line_attr = [-1] * 8
                    if(box['category_name'] == 'zebra_line'):
                        zebra_line_attr[0] = float(min(box_w, max(box['keypoints'][0] - box_tl_x, 0)) / box_w)
                        zebra_line_attr[1] = float(min(box_h, max(box['keypoints'][1] - box_tl_y, 0)) / box_h)
                        zebra_line_attr[2] = float(min(box_w, max(box['keypoints'][3] - box_tl_x, 0)) / box_w)
                        zebra_line_attr[3] = float(min(box_h, max(box['keypoints'][4] - box_tl_y, 0)) / box_h)
                        zebra_line_attr[4] = float(min(box_w, max(box['keypoints'][6] - box_tl_x, 0)) / box_w)
                        zebra_line_attr[5] = float(min(box_h, max(box['keypoints'][7] - box_tl_y, 0)) / box_h)
                        zebra_line_attr[6] = float(min(box_w, max(box['keypoints'][9] - box_tl_x, 0)) / box_w)
                        zebra_line_attr[7] = float(min(box_h, max(box['keypoints'][10] - box_tl_y, 0)) / box_h)

                    # ------ index 69~76: 网格线4个点坐标
                    grid_line_attr = [-1] * 8
                    if(box['category_name'] == 'grid_line'):
                        grid_line_attr[0] = float(min(box_w, max(box['keypoints'][0] - box_tl_x, 0)) / box_w)
                        grid_line_attr[1] = float(min(box_h, max(box['keypoints'][1] - box_tl_y, 0)) / box_h)
                        grid_line_attr[2] = float(min(box_w, max(box['keypoints'][3] - box_tl_x, 0)) / box_w)
                        grid_line_attr[3] = float(min(box_h, max(box['keypoints'][4] - box_tl_y, 0)) / box_h)
                        grid_line_attr[4] = float(min(box_w, max(box['keypoints'][6] - box_tl_x, 0)) / box_w)
                        grid_line_attr[5] = float(min(box_h, max(box['keypoints'][7] - box_tl_y, 0)) / box_h)
                        grid_line_attr[6] = float(min(box_w, max(box['keypoints'][9] - box_tl_x, 0)) / box_w)
                        grid_line_attr[7] = float(min(box_h, max(box['keypoints'][10] - box_tl_y, 0)) / box_h)

                    # ------ index 77~84: 导流线4个点坐标
                    diversion_line_attr = [-1] * 8
                    if(box['category_name'] == 'diversion_line'):
                        diversion_line_attr[0] = float(min(box_w, max(box['keypoints'][0] - box_tl_x, 0)) / box_w)
                        diversion_line_attr[1] = float(min(box_h, max(box['keypoints'][1] - box_tl_y, 0)) / box_h)
                        diversion_line_attr[2] = float(min(box_w, max(box['keypoints'][3] - box_tl_x, 0)) / box_w)
                        diversion_line_attr[3] = float(min(box_h, max(box['keypoints'][4] - box_tl_y, 0)) / box_h)
                        diversion_line_attr[4] = float(min(box_w, max(box['keypoints'][6] - box_tl_x, 0)) / box_w)
                        diversion_line_attr[5] = float(min(box_h, max(box['keypoints'][7] - box_tl_y, 0)) / box_h)
                        diversion_line_attr[6] = float(min(box_w, max(box['keypoints'][9] - box_tl_x, 0)) / box_w)
                        diversion_line_attr[7] = float(min(box_h, max(box['keypoints'][10] - box_tl_y, 0)) / box_h)

                    curr_box_labels = [class_index_0, x_normalized_centre_1, y_normalized_centre_2, w_normalized_3, h_normalized_4] + \
                                        attr_before_lane + \
                                        bus_lane_attr + \
                                        zebra_line_attr + \
                                        grid_line_attr + \
                                        diversion_line_attr
        
                    # 确保标签个数为
                    assert len(curr_box_labels) == 85

                    curr_img_labels.append(curr_box_labels)

            # 写入当前标签进 txt 文件
            if(curr_image_process_done): 
                img_path = image_full_path
                txt_path = img_path[:-3] + 'txt'
                if os.path.exists(img_path):
                    # 如果存在之前的，先删除之前的
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                        print("--->removed label " + txt_path)
                    
                    # 写入新的标签
                    if len(curr_img_labels) != 0:
                        with open(txt_path,'a', encoding='utf-8') as curr_img_labels_file:
                            for box_labels in curr_img_labels:
                                curr_box_written_str = " ".join(str(x) for x in box_labels) + '\n' 
                                curr_img_labels_file.write(curr_box_written_str)
                            print("--->written label " + txt_path)

                        # 把当前文件写入 all.txt
                        with open('all_chezai_dataset_with_attr.txt', 'a', encoding='utf-8') as all_path_files:
                            print(txt_path + "   written in chezai_dataset_with_attr.txt")
                            all_path_files.write(txt_path + '\n')    
                    else:
                        print("no labels!!")
            else:
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
