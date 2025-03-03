# Plotting utils

import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(img.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def plot_wh_methods():  # from utils.plots import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(x, ya, '.-', label='YOLOv3')
    plt.plot(x, yb ** 2, '.-', label='YOLOR ^2')
    plt.plot(x, yb ** 1.6, '.-', label='YOLOR ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    fig.savefig('comparison.png', dpi=200)


def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    # output is (xyxy, conf, cls, attributes_outputs)
    targets = []
    for i, o in enumerate(output):
        for x1, y1, x2, y2, conf, cls, *attributes in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array([x1, y1, x2, y2])[None])), conf, *attributes])
    return np.array(targets)

def plot_images(images, targets, shapes=None, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16, attribute_targets=0, plot_in_original_size=False):
    # Plot image grid with labels

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 1  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    colors = color_list()  # list of colors

    # if dataset image is bigger than the max_size, then 
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)
    
    if plot_in_original_size:
        h = shapes[0][0][0]
        w = shapes[0][0][1]

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, (w, h))

        if plot_in_original_size:
            img = cv2.resize(cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB), (w, h)) 

        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            classes = image_targets[:, 1].astype('int')
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            is_labels = image_targets.shape[1] == (6 + attribute_targets)  # labels if no conf column
            # if is labels: (batch_id, cls, x, y, w, h, attributes_targets)
            # if is preds: (batch_id, cls, x, y, w, h, conf, attributes_outputs)
            conf = None if is_labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01 and not plot_in_original_size:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif plot_in_original_size:
                    if boxes.max() <= 1.01:
                        boxes[[0, 2]] *= images.shape[3]
                        boxes[[1, 3]] *= images.shape[2]
                    # 去除 padding 的偏差
                    dw_padded = shapes[i][1][1][0]
                    boxes[[0, 2]] -= dw_padded
                    w_ori = shapes[i][0][1]
                    w_unpadded = shapes[i][1][0][1] * w_ori
                    boxes[[0, 2]] /= w_unpadded
                    boxes[[0, 2]] *= w
                    dh_padded = shapes[i][1][1][1]
                    boxes[[1, 3]] -= dh_padded
                    h_ori = shapes[i][0][0]
                    h_unpadded = shapes[i][1][0][0] * h_ori
                    boxes[[1, 3]] /= h_unpadded
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor

            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls

                if is_labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if is_labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, img, label=label, color=color, line_thickness=tl)

                attributes_info_str = ""

                ################labels 的属性打印#########################
                if(True):
                    if is_labels and attribute_targets != 0:
                        # for vehicle type
                        if classes[j] == 0:
                            car_type = str(image_targets[j, 6].astype('int'))
                            left_light_seen = str(image_targets[j, 7].astype('int'))
                            left_light_status = str(image_targets[j, 8].astype('int'))
                            left_light_pos_x = image_targets[j, 9].astype('float') * int(box[2] - box[0]) + int(box[0])
                            left_light_pos_y = image_targets[j, 10].astype('float') * int(box[3] - box[1]) + int(box[1])
                            left_light_pos_w = image_targets[j, 11].astype('float') * int(box[2] - box[0]) 
                            left_light_pos_h = image_targets[j, 12].astype('float') * int(box[3] - box[1])
                            if float(left_light_seen) > 0.95:
                                plot_one_box((left_light_pos_x, left_light_pos_y, left_light_pos_x + left_light_pos_w, left_light_pos_y + left_light_pos_h), img, label="left_light", color=color, line_thickness=tl)

                            right_light_seen = str(image_targets[j, 13].astype('int'))
                            right_light_status = str(image_targets[j, 14].astype('int'))
                            right_light_pos_x = image_targets[j, 15].astype('float') * int(box[2] - box[0]) + int(box[0])
                            right_light_pos_y = image_targets[j, 16].astype('float') * int(box[3] - box[1]) + int(box[1])
                            right_light_pos_w = image_targets[j, 17].astype('float') * int(box[2] - box[0]) 
                            right_light_pos_h = image_targets[j, 18].astype('float') * int(box[3] - box[1])
                            if float(right_light_seen) > 0.95:
                                plot_one_box((right_light_pos_x, right_light_pos_y, right_light_pos_x + right_light_pos_w, right_light_pos_y + right_light_pos_h), img, label="right_light", color=color, line_thickness=tl)

                            car_butt_seen = str(image_targets[j, 19].astype('int'))
                            car_butt_pos_x = image_targets[j, 20].astype('float') * int(box[2] - box[0]) + int(box[0])
                            car_butt_pos_y = image_targets[j, 21].astype('float') * int(box[3] - box[1]) + int(box[1])
                            car_butt_pos_w = image_targets[j, 22].astype('float') * int(box[2] - box[0]) 
                            car_butt_pos_h = image_targets[j, 23].astype('float') * int(box[3] - box[1])
                            if float(car_butt_seen) > 0.95:
                                plot_one_box((car_butt_pos_x, car_butt_pos_y, car_butt_pos_x + car_butt_pos_w, car_butt_pos_y + car_butt_pos_h), img, label="car_butt", color=color, line_thickness=tl)

                            car_head_seen = str(image_targets[j, 24].astype('int'))
                            car_head_pos_x = image_targets[j, 25].astype('float') * int(box[2] - box[0]) + int(box[0])
                            car_head_pos_y = image_targets[j, 26].astype('float') * int(box[3] - box[1]) + int(box[1])
                            car_head_pos_w = image_targets[j, 27].astype('float') * int(box[2] - box[0]) 
                            car_head_pos_h = image_targets[j, 28].astype('float') * int(box[3] - box[1])
                            if float(car_head_seen) > 0.95:
                                plot_one_box((car_head_pos_x, car_head_pos_y, car_head_pos_x + car_head_pos_w, car_head_pos_y + car_head_pos_h), img, label="car_head", color=color, line_thickness=tl)

                            car_plate_seen = str(image_targets[j, 29].astype('int'))
                            car_plate_pos_x = image_targets[j, 30].astype('float') * int(box[2] - box[0]) + int(box[0])
                            car_plate_pos_y = image_targets[j, 31].astype('float') * int(box[3] - box[1]) + int(box[1])
                            car_plate_pos_w = image_targets[j, 32].astype('float') * int(box[2] - box[0]) 
                            car_plate_pos_h = image_targets[j, 33].astype('float') * int(box[3] - box[1])
                            if float(car_plate_seen) > 0.95:
                                plot_one_box((car_plate_pos_x, car_plate_pos_y, car_plate_pos_x + car_plate_pos_w, car_plate_pos_y + car_plate_pos_h), img, label="car_plate", color=color, line_thickness=tl)

                            attributes_info_str = car_type + "|" + left_light_seen + left_light_status + "|" + right_light_seen + right_light_status + "|" + car_butt_seen + "|" + car_head_seen + "|" + car_plate_seen

                        # for person type
                        elif classes[j] == 1:
                            person_type = str(image_targets[j, 34].astype('int'))
                            person_status = str(image_targets[j, 35].astype('int'))
                            helmet_state = str(image_targets[j, 36].astype('int'))

                            person_head_seen = str(image_targets[j, 37].astype('int'))
                            person_head_pos_x = image_targets[j, 38].astype('float') * int(box[2] - box[0]) + int(box[0])
                            person_head_pos_y = image_targets[j, 39].astype('float') * int(box[3] - box[1]) + int(box[1])
                            person_head_pos_w = image_targets[j, 40].astype('float') * int(box[2] - box[0]) 
                            person_head_pos_h = image_targets[j, 41].astype('float') * int(box[3] - box[1])
                            if float(person_head_seen) > 0.95:
                                plot_one_box((person_head_pos_x, person_head_pos_y, person_head_pos_x + person_head_pos_w, person_head_pos_y + person_head_pos_h), img, label="person_head", color=color, line_thickness=tl)

                            person_bike_seen = str(image_targets[j, 42].astype('int'))
                            person_bike_pos_x = image_targets[j, 43].astype('float') * int(box[2] - box[0]) + int(box[0])
                            person_bike_pos_y = image_targets[j, 44].astype('float') * int(box[3] - box[1]) + int(box[1])
                            person_bike_pos_w = image_targets[j, 45].astype('float') * int(box[2] - box[0]) 
                            person_bike_pos_h = image_targets[j, 46].astype('float') * int(box[3] - box[1])
                            if float(person_bike_seen) > 0.95:
                                plot_one_box((person_bike_pos_x, person_bike_pos_y, person_bike_pos_x + person_bike_pos_w, person_bike_pos_y + person_bike_pos_h), img, label="person_bike", color=color, line_thickness=tl)
                                
                            bike_plate_seen = str(image_targets[j, 47].astype('int'))
                            bike_plate_pos_x = image_targets[j, 48].astype('float') * int(box[2] - box[0]) + int(box[0])
                            bike_plate_pos_y = image_targets[j, 49].astype('float') * int(box[3] - box[1]) + int(box[1])
                            bike_plate_pos_w = image_targets[j, 50].astype('float') * int(box[2] - box[0]) 
                            bike_plate_pos_h = image_targets[j, 51].astype('float') * int(box[3] - box[1])
                            if float(bike_plate_seen) > 0.95:
                                plot_one_box((bike_plate_pos_x, bike_plate_pos_y, bike_plate_pos_x + bike_plate_pos_w, bike_plate_pos_y + bike_plate_pos_h), img, label="bike_plate", color=color, line_thickness=tl)

                            attributes_info_str = person_type + "|" + person_status + helmet_state + "|" + person_head_seen + "|" + person_bike_seen + "|" + bike_plate_seen

                        # for traffic light type
                        elif classes[j] == 3:
                            traffic_light_status = str(image_targets[j, 52].astype('int'))
                            attributes_info_str = traffic_light_status

                        # for road sign type
                        elif classes[j] == 5:
                            road_sign_status = str(image_targets[j, 53].astype('int'))
                            attributes_info_str = road_sign_status

                        elif classes[j] > 5:
                            if image_targets[j, 54 + (int(classes[j]) - 6) * 8] != -1:
                                p1_x = image_targets[j, 54 + (int(classes[j]) - 6) * 8].astype('float') * int(box[2] - box[0]) + int(box[0])
                                p1_x = min(max(int(p1_x), 0), w)
                                p1_y = image_targets[j, 55 + (int(classes[j]) - 6) * 8].astype('float') * int(box[3] - box[1]) + int(box[1])
                                p1_y = min(max(int(p1_y), 0), h)
                                p2_x = image_targets[j, 56 + (int(classes[j]) - 6) * 8].astype('float') * int(box[2] - box[0]) + int(box[0])
                                p2_x = min(max(int(p2_x), 0), w)
                                p2_y = image_targets[j, 57 + (int(classes[j]) - 6) * 8].astype('float') * int(box[3] - box[1]) + int(box[1])
                                p2_y = min(max(int(p2_y), 0), h)
                                p3_x = image_targets[j, 58 + (int(classes[j]) - 6) * 8].astype('float') * int(box[2] - box[0]) + int(box[0])
                                p3_x = min(max(int(p3_x), 0), w)
                                p3_y = image_targets[j, 59 + (int(classes[j]) - 6) * 8].astype('float') * int(box[3] - box[1]) + int(box[1])
                                p3_y = min(max(int(p3_y), 0), h)
                                p4_x = image_targets[j, 60 + (int(classes[j]) - 6) * 8].astype('float') * int(box[2] - box[0]) + int(box[0])
                                p4_x = min(max(int(p4_x), 0), w)
                                p4_y = image_targets[j, 61 + (int(classes[j]) - 6) * 8].astype('float') * int(box[3] - box[1]) + int(box[1])
                                p4_y = min(max(int(p4_y), 0), h)
                                cv2.fillPoly(img, [np.array([(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y), (p4_x, p4_y)], np.int32)], color)
                            
                    ############## pred 属性的打印 ################################
                    elif not is_labels and attribute_targets != 0:
                        # for vehicle type
                        if classes[j] == 0:
                            car_type = str(image_targets[j, 7].astype('int'))
                            car_type_conf = str(image_targets[j, 8].astype('float'))[:4]

                            left_light_seen_conf = str(round(image_targets[j, 10].astype('float'), 2))
                            left_light_status = str(image_targets[j, 11].astype('int'))
                            left_light_status_conf = str(round(image_targets[j, 12].astype('float'), 2))
                            left_light_pos_x = image_targets[j, 13].astype('float') * int(box[2] - box[0]) + int(box[0])
                            left_light_pos_y = image_targets[j, 14].astype('float') * int(box[3] - box[1]) + int(box[1])
                            left_light_pos_w = image_targets[j, 15].astype('float') * int(box[2] - box[0]) 
                            left_light_pos_h = image_targets[j, 16].astype('float') * int(box[3] - box[1])
                            if float(left_light_seen_conf) > 0.95:
                                plot_one_box((left_light_pos_x, left_light_pos_y, left_light_pos_x + left_light_pos_w, left_light_pos_y + left_light_pos_h), img, label="left_light", color=color, line_thickness=tl)

                            right_light_seen_conf = str(round(image_targets[j, 17].astype('float'), 2))
                            right_light_status = str(image_targets[j, 18].astype('int'))
                            right_light_status_conf = str(round(image_targets[j, 19].astype('float'), 2))
                            right_light_pos_x = image_targets[j, 20].astype('float') * int(box[2] - box[0]) + int(box[0])
                            right_light_pos_y = image_targets[j, 21].astype('float') * int(box[3] - box[1]) + int(box[1])
                            right_light_pos_w = image_targets[j, 22].astype('float') * int(box[2] - box[0]) 
                            right_light_pos_h = image_targets[j, 23].astype('float') * int(box[3] - box[1])
                            if float(right_light_seen_conf) > 0.95:
                                plot_one_box((right_light_pos_x, right_light_pos_y, right_light_pos_x + right_light_pos_w, right_light_pos_y + right_light_pos_h), img, label="right_light", color=color, line_thickness=tl)

                            car_butt_seen_conf = str(round(image_targets[j, 24].astype('float'), 2))
                            car_butt_pos_x = image_targets[j, 25].astype('float') * int(box[2] - box[0]) + int(box[0])
                            car_butt_pos_y = image_targets[j, 26].astype('float') * int(box[3] - box[1]) + int(box[1])
                            car_butt_pos_w = image_targets[j, 27].astype('float') * int(box[2] - box[0]) 
                            car_butt_pos_h = image_targets[j, 28].astype('float') * int(box[3] - box[1])
                            if float(car_butt_seen_conf) > 0.95:
                                plot_one_box((car_butt_pos_x, car_butt_pos_y, car_butt_pos_x + car_butt_pos_w, car_butt_pos_y + car_butt_pos_h), img, label="car_butt", color=color, line_thickness=tl)
                            
                            car_head_seen_conf = str(round(image_targets[j, 29].astype('float'), 2))
                            car_head_pos_x = image_targets[j, 30].astype('float') * int(box[2] - box[0]) + int(box[0])
                            car_head_pos_y = image_targets[j, 31].astype('float') * int(box[3] - box[1]) + int(box[1])
                            car_head_pos_w = image_targets[j, 32].astype('float') * int(box[2] - box[0]) 
                            car_head_pos_h = image_targets[j, 33].astype('float') * int(box[3] - box[1])
                            if float(car_head_seen_conf) > 0.95:
                                plot_one_box((car_head_pos_x, car_head_pos_y, car_head_pos_x + car_head_pos_w, car_head_pos_y + car_head_pos_h), img, label="car_head", color=color, line_thickness=tl)

                            car_plate_seen_conf = str(round(image_targets[j, 34].astype('float'), 2))
                            car_plate_pos_x = image_targets[j, 35].astype('float') * int(box[2] - box[0]) + int(box[0])
                            car_plate_pos_y = image_targets[j, 36].astype('float') * int(box[3] - box[1]) + int(box[1])
                            car_plate_pos_w = image_targets[j, 37].astype('float') * int(box[2] - box[0]) 
                            car_plate_pos_h = image_targets[j, 38].astype('float') * int(box[3] - box[1])
                            if float(car_plate_seen_conf) > 0.95:
                                plot_one_box((car_plate_pos_x, car_plate_pos_y, car_plate_pos_x + car_plate_pos_w, car_plate_pos_y + car_plate_pos_h), img, label="car_plate", color=color, line_thickness=tl)

                            attributes_info_str = car_type + "(" + car_type_conf + ")" + "|" + left_light_seen_conf + "->" + left_light_status + "(" + left_light_status_conf + ")" + "|" + right_light_seen_conf + "->" + right_light_status + "(" + right_light_status_conf + ")" + "|" + car_butt_seen_conf + "|" + car_head_seen_conf + "|" + car_plate_seen_conf;

                        # for person type
                        if classes[j] == 1:
                            person_type = str(image_targets[j, 39].astype('int'))
                            person_type_conf = str(round(image_targets[j, 40].astype('float'), 2))
                            person_status = str(image_targets[j, 41].astype('int'))
                            person_status_conf = str(round(image_targets[j, 42].astype('float'), 2))
                            helmet_state = str(image_targets[j, 43].astype('int'))
                            helmet_state_conf = str(round(image_targets[j, 44].astype('float'), 2))

                            person_head_seen_conf = str(round(image_targets[j, 45].astype('float'), 2))
                            person_head_pos_x = image_targets[j, 46].astype('float') * int(box[2] - box[0]) + int(box[0])
                            person_head_pos_y = image_targets[j, 47].astype('float') * int(box[3] - box[1]) + int(box[1])
                            person_head_pos_w = image_targets[j, 48].astype('float') * int(box[2] - box[0]) 
                            person_head_pos_h = image_targets[j, 49].astype('float') * int(box[3] - box[1])
                            if float(person_head_seen_conf) > 0.95:
                                plot_one_box((person_head_pos_x, person_head_pos_y, person_head_pos_x + person_head_pos_w, person_head_pos_y + person_head_pos_h), img, label="person_head", color=color, line_thickness=tl)

                            person_bike_seen_conf = str(round(image_targets[j, 50].astype('float'), 2))
                            person_bike_pos_x = image_targets[j, 51].astype('float') * int(box[2] - box[0]) + int(box[0])
                            person_bike_pos_y = image_targets[j, 52].astype('float') * int(box[3] - box[1]) + int(box[1])
                            person_bike_pos_w = image_targets[j, 53].astype('float') * int(box[2] - box[0]) 
                            person_bike_pos_h = image_targets[j, 54].astype('float') * int(box[3] - box[1])
                            if float(person_bike_seen_conf) > 0.95:
                                plot_one_box((person_bike_pos_x, person_bike_pos_y, person_bike_pos_x + person_bike_pos_w, person_bike_pos_y + person_bike_pos_h), img, label="person_bike", color=color, line_thickness=tl)

                            bike_plate_seen_conf = str(round(image_targets[j, 55].astype('float'), 2))
                            bike_plate_pos_x = image_targets[j, 56].astype('float') * int(box[2] - box[0]) + int(box[0])
                            bike_plate_pos_y = image_targets[j, 57].astype('float') * int(box[3] - box[1]) + int(box[1])
                            bike_plate_pos_w = image_targets[j, 58].astype('float') * int(box[2] - box[0]) 
                            bike_plate_pos_h = image_targets[j, 59].astype('float') * int(box[3] - box[1])
                            if float(bike_plate_seen_conf) > 0.95:
                                plot_one_box((bike_plate_pos_x, bike_plate_pos_y, bike_plate_pos_x + bike_plate_pos_w, bike_plate_pos_y + bike_plate_pos_h), img, label="bike_plate", color=color, line_thickness=tl)

                            attributes_info_str = person_type + "(" + person_type_conf + ")" + "|" + person_status + "(" + person_status_conf + ")" + "|" + helmet_state + "(" + helmet_state_conf + ")" + "|" + person_head_seen_conf + "|" + person_bike_seen_conf + "|" + bike_plate_seen_conf

                        # for traffic light type
                        if classes[j] == 3:
                            traffic_light_status = str(image_targets[j, 60].astype('int'))
                            traffic_light_status_conf = str(round(image_targets[j, 61].astype('float'), 2))
                            attributes_info_str = traffic_light_status + "(" + traffic_light_status_conf + ")" 

                        # for road sign type
                        if classes[j] == 5:
                            road_sign_status = str(image_targets[j, 63].astype('int'))
                            road_sign_status_conf = str(round(image_targets[j, 64].astype('float'), 2))
                            attributes_info_str = road_sign_status + "(" + road_sign_status_conf + ")" 

                        if classes[j] > 5:
                            if image_targets[j, 65 + (int(classes[j]) - 6) * 8] != -1:
                                p1_x = image_targets[j, 65 + (int(classes[j]) - 6) * 8].astype('float') * int(box[2] - box[0]) + int(box[0])
                                p1_x = min(max(int(p1_x), 0), w)
                                p1_y = image_targets[j, 66 + (int(classes[j]) - 6) * 8].astype('float') * int(box[3] - box[1]) + int(box[1])
                                p1_y = min(max(int(p1_y), 0), h)
                                p2_x = image_targets[j, 67 + (int(classes[j]) - 6) * 8].astype('float') * int(box[2] - box[0]) + int(box[0])
                                p2_x = min(max(int(p2_x), 0), w)
                                p2_y = image_targets[j, 68 + (int(classes[j]) - 6) * 8].astype('float') * int(box[3] - box[1]) + int(box[1])
                                p2_y = min(max(int(p2_y), 0), h)
                                p3_x = image_targets[j, 69 + (int(classes[j]) - 6) * 8].astype('float') * int(box[2] - box[0]) + int(box[0])
                                p3_x = min(max(int(p3_x), 0), w)
                                p3_y = image_targets[j, 70 + (int(classes[j]) - 6) * 8].astype('float') * int(box[3] - box[1]) + int(box[1])
                                p3_y = min(max(int(p3_y), 0), h)
                                p4_x = image_targets[j, 71 + (int(classes[j]) - 6) * 8].astype('float') * int(box[2] - box[0]) + int(box[0])
                                p4_x = min(max(int(p4_x), 0), w)
                                p4_y = image_targets[j, 72 + (int(classes[j]) - 6) * 8].astype('float') * int(box[3] - box[1]) + int(box[1])
                                p4_y = min(max(int(p4_y), 0), h)
                                cv2.fillPoly(img, [np.array([(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y), (p4_x, p4_y)], np.int32)], color)

                t_size = cv2.getTextSize(attributes_info_str, 0, fontScale=tl/3, thickness=tf)[0]
                c1 = (int(box[0]), int(box[3]))
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, attributes_info_str, (c1[0], c1[1] - 2), 0, tl/3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(img, label, (5, t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), thickness=3)

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))
        mosaic[block_y : block_y + h, block_x : block_x + w, :] = img

    if fname:
        #r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        #mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname.as_posix(), cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save

    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_test_txt():  # from utils.plots import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_study_txt(path='', x=None):  # from utils.plots import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    # ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [Path(path) / f'study_coco_{x}.txt' for x in ['yolor-p6', 'yolor-w6', 'yolor-e6', 'yolor-d6']]:
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        # for i in range(7):
        #     ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
        #     ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)


def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
    # plot dataset labels
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    colors = color_list()
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors[int(cls) % 10])  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

    # loggers
    for k, v in loggers.items() or {}:
        if k == 'wandb' and v:
            v.log({"Labels": [v.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=False)


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):  # from utils.plots import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    # Plot iDetection '*.txt' per-image logs. from utils.plots import *; profile_idetection()
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.plots import *; plot_results_overlay()
    # Plot training 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']  # legends
    t = ['Box', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
                # y_smooth = butter_lowpass_filtfilt(y)
                # ax[i].plot(x, np.gradient(y_smooth), marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir=''):
    # Plot training 'results*.txt'. from utils.plots import *; plot_results(save_dir='runs/train/exp')
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        # files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
        files = ['results%g.txt' % x for x in id]
        c = ('gsutil cp ' + '%s ' * len(files) + '.') % tuple('gs://%s/results%g.txt' % (bucket, x) for x in id)
        os.system(c)
    else:
        files = list(Path(save_dir).glob('results*.txt'))
    assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else f.stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)
    
    
def output_to_keypoint(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.detach().cpu().numpy()[index])])
    return np.array(targets)


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
