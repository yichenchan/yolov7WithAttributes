import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
import time
import math

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.25,
         iou_thres=0.45,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         plot_in_original_size=False
         ):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if trace:
            model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    attribute_targets = data['attribute_targets'] if "attribute_targets" in data.keys() else 0
    attribute_outputs = data['attribute_outputs'] if "attribute_outputs" in data.keys() else 0
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, attribute_targets, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')

    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    car_total = 0.00001
    car_type_correct = 0.00001
    car_light_total = 0.00001
    car_light_visable_correct = 0.00001
    car_light_on_correct = 0.00001
    car_light_pos_correct = 0.00001
    car_plate_total = 0.00001
    car_plate_visabel_correct = 0.00001 
    car_plate_pos_correct = 0.00001 
    car_butt_total = 0.00001
    car_butt_visabel_correct = 0.00001 
    car_butt_pos_correct = 0.00001 
    car_head_total = 0.00001
    car_head_visable_correct = 0.00001
    car_head_pos_correct = 0.00001
    person_total = 0.00001
    person_type_correct = 0.00001
    person_status_correct = 0.00001
    person_helmet_correct = 0.00001
    person_head_total = 0.00001
    person_head_visabel_correct = 0.00001
    person_head_pos_correct = 0.00001
    person_bike_total = 0.00001
    person_bike_visable_correct = 0.00001
    person_bike_pos_correct = 0.00001
    person_bikeplate_total = 0.00001
    person_bikeplate_visabel_correct = 0.00001
    person_bikeplate_pos_correct = 0.00001
    traffic_light_total = 0.00001
    traffic_light_type_correct = 0.00001
    road_sign_total = 0.00001
    road_sign_type_correct = 0.00001
    bus_lane_total = 0.00001
    bus_lane_points_dist_total = 0.0000001
    zebra_line_total = 0.00001
    zebra_line_points_dist_total = 0.00001
    grid_line_total = 0.00001
    grid_line_points_dist_total = 0.00001
    diversion_line_total = 0.00001
    diversion_line_points_dist_total = 0.00001

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            # print("out", out.shape)
            # print("train_out[0]", train_out[0].shape)
            # print("train_out[1]", train_out[1].shape)
            # print("train_out[2]", train_out[2].shape)
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:6] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=False, attribute_outputs=attribute_outputs)
            # out is (xyxy, conf, cls, attributes)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # restore Predictions to original size
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            # if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
            #     if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
            #         box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
            #                      "class_id": int(cls),
            #                      "box_caption": "%s %.3f" % (names[cls], conf),
            #                      "scores": {"class_score": conf},
            #                      "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
            #         boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
            #         wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            # wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # restore target boxes to the original size
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # 获取和 pred 里每个元素 iou 最大的对应 label index
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        # 遍历当前 cls 所有 iou > 0.5 的 pred 对应的 label 在 i 中的索引值
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # 遍历到的 label index
                            p = pi[j]  # 遍历到的 pred index

                            if d.item() not in detected_set:

                                # 找到匹配的 target 和 label，开始计算 attribute metric
                                if attribute_outputs != 0 and nl:
                                    label_attributions = labels[d, 5:][0]
                                    # 车辆类别（0），
                                    # 左车灯是否可见（1），左车灯状态（2），左车灯位置（3~6），
                                    # 右车灯是否可见（7），右车灯状态（8），右车灯位置（9~12），
                                    # 车屁股是否可见（13），车屁股框位置（14~17）
                                    # 车头是否可见（18），车头框位置（19~22）
                                    # 车牌是否可见（23），车牌位置（24~27），
                                    # 人的类别（28），人的状态（29）， 是否带头盔（30）
                                    # 人头是否可见（31），人头的位置（32~35），
                                    # 人骑的车是否可见（36），人骑的车的位置（37~40）
                                    # 电动车车牌是否可见（41），车牌的位置（42~45），
                                    # 红绿灯的颜色（46），
                                    # 路牌的颜色（47），
                                    # 公交车道第一个点坐标（48,49），公交车道第二个点坐标（50,51），公交车道第三个点坐标（52,53），公交车道第四个点坐标（54,55），
                                    # 斑马线第一个点坐标（56,57），斑马线第二个点坐标（58,59），斑马线第三个点坐标（60,61），斑马线第四个点坐标（62,63），
                                    # 网格线第一个点坐标（64,65），网格线第二个点坐标（66,67），网格线第三个点坐标（68.69），网格线第四个点坐标（70,71），
                                    # 导流线第一个点坐标（72,73），导流线第二个点坐标（74,75），导流线第三个点坐标（76,77），导流线第四个点坐标（78,79）
                                    pred_attributions = pred[p, 6:][0]
                                    ## pred_attributions : 
                                    # 车辆类别（0），车辆类别置信度(1), 无意义值（2），
                                    # 左车灯是否可见置信度（3），左车灯状态（4），左车灯状态置信度（5），左车灯位置（6~9），
                                    # 右车灯是否可见置信度（10），右车灯状态（11），右车灯状态置信度（12），右车灯位置（13~16），
                                    # 车屁股是否可见置信度（17），车屁股框位置（18~21）
                                    # 车头是否可见置信度（22），车头框位置（23~26）
                                    # 车牌是否可见置信度（27），车牌位置（28~31），
                                    # 人的类别（32），人的类别置信度（33），人的状态（34），人的状态置信度（35），是否带头盔（36），是否带头盔置信度（37），
                                    # 人头是否可见置信度（38），人头的位置（39~42），
                                    # 人骑的车是否可见置信度（43），人骑的车的位置（44~47）
                                    # 电动车车牌是否可见置信度（48），车牌的位置（49~52），
                                    # 红绿灯的颜色（53），红绿灯的颜色置信度（54），无意义值（55），
                                    # 路牌的颜色（56），路牌的置信度（57），
                                    # 公交车道第一个点坐标（58,59），公交车道第二个点坐标（60,61），公交车道第三个点坐标（62,63），公交车道第四个点坐标（64,65），
                                    # 斑马线第一个点坐标（66,67），斑马线第二个点坐标（68,69），斑马线第三个点坐标（70,71），斑马线第四个点坐标（72,73），
                                    # 网格线第一个点坐标（74,75），网格线第二个点坐标（76,77），网格线第三个点坐标（78.79），网格线第四个点坐标（80,81），
                                    # 导流线第一个点坐标（82,83），导流线第二个点坐标（84,85），导流线第三个点坐标（86,87），导流线第四个点坐标（88,89）
                                    visable_thresh = 0.95
                                    light_iou_thresh = 0.5
                                    plate_iou_thresh = 0.7
                                    car_butt_iou_thresh = 0.5
                                    car_head_iou_thresh = 0.5
                                    if cls == 0:
                                        car_total += 1
                                        if label_attributions[0] == pred_attributions[0]:
                                            car_type_correct += 1
                                        if label_attributions[1] == 1:
                                            car_light_total += 1
                                            if pred_attributions[3] > visable_thresh:
                                                car_light_visable_correct += 1
                                            if pred_attributions[4] == label_attributions[2]:
                                                car_light_on_correct += 1
                                            left_light_label_box = torch.FloatTensor([[label_attributions[3], label_attributions[4], label_attributions[3] + label_attributions[5], label_attributions[4] + label_attributions[6]]])
                                            left_light_pred_box = torch.FloatTensor([[pred_attributions[6], pred_attributions[7], pred_attributions[6] + pred_attributions[8], pred_attributions[7] + pred_attributions[9]]])
                                            if(box_iou(left_light_label_box, left_light_pred_box) > light_iou_thresh):
                                                car_light_pos_correct += 1

                                        if label_attributions[7] == 1:
                                            car_light_total += 1
                                            if pred_attributions[10] > visable_thresh:
                                                car_light_visable_correct += 1
                                            if pred_attributions[11] == label_attributions[8]:
                                                car_light_on_correct += 1
                                            right_light_label_box = torch.FloatTensor([[label_attributions[9], label_attributions[10], label_attributions[9] + label_attributions[11], label_attributions[10] + label_attributions[12]]])
                                            right_light_pred_box = torch.FloatTensor([[pred_attributions[13], pred_attributions[14], pred_attributions[13] + pred_attributions[15], pred_attributions[14] + pred_attributions[16]]])
                                            if(box_iou(right_light_label_box, right_light_pred_box) > light_iou_thresh):
                                                car_light_pos_correct += 1

                                        if label_attributions[13] == 1:
                                            car_butt_total += 1
                                            if pred_attributions[17] > visable_thresh:
                                                car_butt_visabel_correct += 1
                                                car_butt_label_box = torch.FloatTensor([[label_attributions[14], label_attributions[15], label_attributions[14] + label_attributions[16], label_attributions[15] + label_attributions[17]]])
                                                car_butt_pred_box = torch.FloatTensor([[pred_attributions[18], pred_attributions[19], pred_attributions[18] + pred_attributions[20], pred_attributions[19] + pred_attributions[21]]])
                                                if(box_iou(car_butt_label_box, car_butt_pred_box) > car_butt_iou_thresh):
                                                    car_butt_pos_correct += 1

                                        if label_attributions[18] == 1:
                                            car_head_total += 1
                                            if pred_attributions[22] > visable_thresh:
                                                car_head_visable_correct += 1
                                                car_head_label_box = torch.FloatTensor([[label_attributions[19], label_attributions[20], label_attributions[19] + label_attributions[21], label_attributions[20] + label_attributions[22]]])
                                                car_head_pred_box = torch.FloatTensor([[pred_attributions[23], pred_attributions[24], pred_attributions[23] + pred_attributions[25], pred_attributions[24] + pred_attributions[26]]])
                                                if(box_iou(car_head_label_box, car_head_pred_box) > car_head_iou_thresh):
                                                    car_head_pos_correct += 1

                                        if label_attributions[23] == 1:
                                            car_plate_total += 1
                                            if pred_attributions[27] > visable_thresh:
                                                car_plate_visabel_correct += 1
                                                car_plate_label_box = torch.FloatTensor([[label_attributions[24], label_attributions[25], label_attributions[24] + label_attributions[26], label_attributions[25] + label_attributions[27]]])
                                                car_plate_pred_box = torch.FloatTensor([[pred_attributions[28], pred_attributions[29], pred_attributions[28] + pred_attributions[30], pred_attributions[29] + pred_attributions[31]]])
                                                if(box_iou(car_plate_label_box, car_plate_pred_box) > plate_iou_thresh):
                                                    car_plate_pos_correct += 1
                                    
                                    head_iou_thresh = 0.5
                                    bike_iou_thresh = 0.5
                                    bikeplate_iou_thresh = 0.5
                                    if cls == 1:
                                        person_total += 1
                                        if label_attributions[28] == pred_attributions[32]:
                                            person_type_correct += 1
                                        if label_attributions[29] == pred_attributions[34]:
                                            person_status_correct += 1
                                        if label_attributions[30] == pred_attributions[36]:
                                            person_helmet_correct += 1
                                        
                                        if label_attributions[31] == 1:
                                            person_head_total += 1
                                            if pred_attributions[38] > visable_thresh:
                                                person_head_visabel_correct += 1
                                            person_head_label_box = torch.FloatTensor([[label_attributions[32], label_attributions[33], label_attributions[32] + label_attributions[34], label_attributions[33] + label_attributions[35]]])
                                            person_head_pred_box = torch.FloatTensor([[pred_attributions[39], pred_attributions[40], pred_attributions[39] + pred_attributions[41], pred_attributions[40] + pred_attributions[42]]])
                                            if(box_iou(person_head_label_box, person_head_pred_box) > head_iou_thresh):
                                                person_head_pos_correct += 1

                                        if label_attributions[36] == 1:
                                            person_bike_total += 1
                                            if pred_attributions[43] > visable_thresh:
                                                person_bike_visable_correct += 1
                                            person_bike_label_box = torch.FloatTensor([[label_attributions[37], label_attributions[38], label_attributions[37] + label_attributions[39], label_attributions[38] + label_attributions[40]]])
                                            person_bike_pred_box = torch.FloatTensor([[pred_attributions[44], pred_attributions[45], pred_attributions[44] + pred_attributions[46], pred_attributions[45] + pred_attributions[47]]])
                                            if(box_iou(person_bike_label_box, person_bike_pred_box) > bike_iou_thresh):
                                                person_bike_pos_correct += 1

                                        if label_attributions[41] == 1:
                                            person_bikeplate_total += 1
                                            if pred_attributions[48] > visable_thresh:
                                                person_bikeplate_visabel_correct += 1
                                            person_bikeplate_label_box = torch.FloatTensor([[label_attributions[42], label_attributions[43], label_attributions[42] + label_attributions[44], label_attributions[43] + label_attributions[45]]])
                                            person_bikeplate_pred_box = torch.FloatTensor([[pred_attributions[49], pred_attributions[50], pred_attributions[49] + pred_attributions[51], pred_attributions[50] + pred_attributions[52]]])
                                            if(box_iou(person_bikeplate_label_box, person_bikeplate_pred_box) > bikeplate_iou_thresh):
                                                    person_bikeplate_pos_correct += 1

                                    if cls == 3:
                                        traffic_light_total += 1
                                        if label_attributions[46] == pred_attributions[53]:
                                            traffic_light_type_correct += 1

                                    if cls == 5:
                                        road_sign_total += 1
                                        if label_attributions[47] == pred_attributions[56]:
                                            road_sign_type_correct += 1

                                    if cls == 6:
                                        bus_lane_total += 1
                                        p1_dist = math.sqrt((pred_attributions[58] - label_attributions[48]) ** 2 + (pred_attributions[59] - label_attributions[49]) ** 2)
                                        p2_dist = math.sqrt((pred_attributions[60] - label_attributions[50]) ** 2 + (pred_attributions[61] - label_attributions[51]) ** 2)
                                        p3_dist = math.sqrt((pred_attributions[62] - label_attributions[52]) ** 2 + (pred_attributions[63] - label_attributions[53]) ** 2)
                                        p4_dist = math.sqrt((pred_attributions[64] - label_attributions[54]) ** 2 + (pred_attributions[65] - label_attributions[55]) ** 2)
                                        bus_lane_points_dist_total += ((p1_dist + p2_dist + p3_dist + p4_dist) / 4)

                                    if cls == 7:
                                        zebra_line_total += 1
                                        p1_dist = math.sqrt((pred_attributions[66] - label_attributions[56]) ** 2 + (pred_attributions[67] - label_attributions[57]) ** 2)
                                        p2_dist = math.sqrt((pred_attributions[68] - label_attributions[58]) ** 2 + (pred_attributions[69] - label_attributions[59]) ** 2)
                                        p3_dist = math.sqrt((pred_attributions[70] - label_attributions[60]) ** 2 + (pred_attributions[71] - label_attributions[61]) ** 2)
                                        p4_dist = math.sqrt((pred_attributions[72] - label_attributions[62]) ** 2 + (pred_attributions[73] - label_attributions[63]) ** 2)
                                        zebra_line_points_dist_total += ((p1_dist + p2_dist + p3_dist + p4_dist) / 4)

                                    if cls == 8:
                                        grid_line_total += 1
                                        p1_dist = math.sqrt((pred_attributions[74] - label_attributions[64]) ** 2 + (pred_attributions[75] - label_attributions[65]) ** 2)
                                        p2_dist = math.sqrt((pred_attributions[76] - label_attributions[66]) ** 2 + (pred_attributions[77] - label_attributions[67]) ** 2)
                                        p3_dist = math.sqrt((pred_attributions[78] - label_attributions[68]) ** 2 + (pred_attributions[79] - label_attributions[69]) ** 2)
                                        p4_dist = math.sqrt((pred_attributions[80] - label_attributions[70]) ** 2 + (pred_attributions[81] - label_attributions[71]) ** 2)
                                        grid_line_points_dist_total += ((p1_dist + p2_dist + p3_dist + p4_dist) / 4)

                                    if cls == 9:
                                        diversion_line_total += 1
                                        p1_dist = math.sqrt((pred_attributions[82] - label_attributions[72]) ** 2 + (pred_attributions[83] - label_attributions[73]) ** 2)
                                        p2_dist = math.sqrt((pred_attributions[84] - label_attributions[74]) ** 2 + (pred_attributions[85] - label_attributions[75]) ** 2)
                                        p3_dist = math.sqrt((pred_attributions[86] - label_attributions[76]) ** 2 + (pred_attributions[87] - label_attributions[77]) ** 2)
                                        p4_dist = math.sqrt((pred_attributions[88] - label_attributions[78]) ** 2 + (pred_attributions[89] - label_attributions[79]) ** 2)
                                        diversion_line_points_dist_total += ((p1_dist + p2_dist + p3_dist + p4_dist) / 4)

                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, shapes, paths, f, names, imgsz, 16, attribute_targets, plot_in_original_size), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), shapes, paths, f, names, imgsz, 16, attribute_targets, plot_in_original_size), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # 打印 attribute metrics
    carAttriPf = '%14s' * 10
    print(carAttriPf % ('carTypeP', 'carLightVisP', 'carLightOnP', 'carLightPosP', 'carPlateVisP', 'carPlatePosP', 'carButtVisP', 'carButtPosP', 'carHeadVisP', 'carHeadPosP'))
    carTypeP = round(float(car_type_correct / car_total), 2)
    carLightVisP = round(float(car_light_visable_correct / car_light_total), 2)
    carLightOnP = round(float(car_light_on_correct / car_light_total), 2)
    carLightPosP = round(float(car_light_pos_correct / car_light_total), 2)
    carPlateVisP = round(float(car_plate_visabel_correct / car_plate_total), 2)
    carPlatePosP = round(float(car_plate_pos_correct / car_plate_total), 2)
    carButtVisP = round(float(car_butt_visabel_correct / car_butt_total), 2)
    carButtPosP = round(float(car_butt_pos_correct / car_butt_total), 2)
    carHeadVisP = round(float(car_head_visable_correct / car_head_total), 2)
    carHeadPosP = round(float(car_head_pos_correct / car_head_total), 2)
    print(carAttriPf % (str(carTypeP)+'('+str(int(car_total))+')', \
                        str(carLightVisP)+'('+str(int(car_light_total))+')', str(carLightOnP)+'('+str(int(car_light_total))+')', str(carLightPosP)+'('+str(int(car_light_total))+')', \
                        str(carPlateVisP)+'('+str(int(car_plate_total))+')', str(carPlatePosP)+'('+str(int(car_plate_total))+')', \
                        str(carButtVisP)+'('+str(int(car_butt_total))+')', str(carButtPosP)+'('+str(int(car_butt_total))+')', \
                        str(carHeadVisP)+'('+str(int(car_head_total))+')', str(carHeadPosP)+'('+str(int(car_head_total))+')'))

    personAttrPf = '%14s' * 9
    print(personAttrPf % ('perTypeP', 'perStatusP', 'perHelmetP', 'perHeadVisP', 'perHeadPosP', 'perBikeVisP', 'perBikePosP', 'bikePlateVisP', 'bikePlatePosP'))
    perTypeP = round(float(person_type_correct / person_total), 2)
    perStatusP = round(float(person_status_correct / person_total), 2)
    perHelmetP = round(float(person_helmet_correct / person_total), 2)
    perHeadVisP = round(float(person_head_visabel_correct / person_head_total), 2)
    perHeadPosP = round(float(person_head_pos_correct / person_head_total), 2)
    perBikeVisP = round(float(person_bike_visable_correct / person_bike_total), 2)
    perBikePosP = round(float(person_bike_pos_correct / person_bike_total), 2)
    bikePlateVisP = round(float(person_bikeplate_visabel_correct / person_bikeplate_total), 2)
    bikePlatePosP = round(float(person_bikeplate_pos_correct / person_bikeplate_total), 2)
    print(personAttrPf % (str(perTypeP)+'('+str(int(person_total))+')', str(perStatusP)+'('+str(int(person_total))+')', str(perHelmetP)+'('+str(int(person_total))+')', \
                          str(perHeadVisP)+'('+str(int(person_head_total))+')', str(perHeadPosP)+'('+str(int(person_head_total))+')', \
                          str(perBikeVisP)+'('+str(int(person_bike_total))+')', str(perBikePosP)+'('+str(int(person_bike_total))+')', \
                          str(bikePlateVisP)+'('+str(int(person_bikeplate_total))+')', str(bikePlatePosP)+'('+str(int(person_bikeplate_total))+')'))

    othersAttrPf = '%22s' * 6
    print(othersAttrPf % ('trafficLightColorP', 'roadSignColorP', 'busLaneMeanDist', 'zebraLineMeanDist', 'gridLineMeanDist', 'divLineMeanDist'))
    trafficLightColorP = round(float(traffic_light_type_correct / traffic_light_total), 2)
    roadSignColorP = round(float(road_sign_type_correct / road_sign_total), 2)
    busLaneMeanDist = round(float(bus_lane_points_dist_total / bus_lane_total), 2)
    zebraLineMeanDist = round(float(zebra_line_points_dist_total / zebra_line_total), 2)
    gridLineMeanDist = round(float(grid_line_points_dist_total / grid_line_total), 2)
    divLineMeanDist = round(float(diversion_line_points_dist_total / diversion_line_total), 2)
    print(othersAttrPf % (str(trafficLightColorP)+'('+str(int(traffic_light_total))+')', str(roadSignColorP)+'('+str(int(road_sign_total))+')', \
                          str(busLaneMeanDist)+'('+str(int(bus_lane_total))+')', \
                          str(zebraLineMeanDist)+'('+str(int(zebra_line_total))+')', \
                          str(gridLineMeanDist)+'('+str(int(grid_line_total))+')', \
                          str(divLineMeanDist)+'('+str(int(diversion_line_total))+')'))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--plot-in-original-size', action='store_true')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             plot_in_original_size=opt.plot_in_original_size
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
