from PIL import Image
import os
import onnxruntime as ort
import numpy as np
import cv2
import torch
import time
import torchvision

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, attribute_outputs=0,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5 - attribute_outputs # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    # output be like (xyxy, conf, cls, attributes)
    output = [torch.zeros((0, 6 + attribute_outputs), device=prediction.device)] * prediction.shape[0] 

    # prediction be like (batch_size, na*ny*nx, no)
    for xi, x in enumerate(prediction):  # image index, image inference
        # xi(0~batch_size) is the index of image in a batch
        # x be like (na*ny*nx, (xywh, obj, cls_one-hot, attribute_outputs))

        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5 : nc] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5 : nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box_xywh = x[:, :4]
        box = xywh2xyxy(x[:, :4])

        # attributes
        attributes = x[:, 5 + nc : 5 + nc + attribute_outputs]
        if attribute_outputs != 0:
            # 占位符
            place_holder = torch.full_like(attributes[:, 0:1], -1)

            car_type_conf, car_type = attributes[:, 0:3].max(1, keepdims=True)
            car_left_light_seen_conf = attributes[:, 3:4]
            car_left_light_status_conf, car_left_light_status = attributes[:, 4:6].max(1, keepdims=True)
            car_left_light_x = attributes[:, 6:7] 
            car_left_light_y = attributes[:, 7:8] 
            car_left_light_w = attributes[:, 8:9] 
            car_left_light_h = attributes[:, 9:10]
            car_right_light_seen_conf = attributes[:, 10:11]
            car_right_light_status_conf, car_right_light_status = attributes[:, 11:13].max(1, keepdims=True)
            car_right_light_x = attributes[:, 13:14] 
            car_right_light_y = attributes[:, 14:15] 
            car_right_light_w = attributes[:, 15:16] 
            car_right_light_h = attributes[:, 16:17] 
            car_butt_seen_conf = attributes[:, 17:18]
            car_butt_x = attributes[:, 18:19] 
            car_butt_y = attributes[:, 19:20] 
            car_butt_w = attributes[:, 20:21] 
            car_butt_h = attributes[:, 21:22] 
            car_head_seen_conf = attributes[:, 22:23]
            car_head_x = attributes[:, 23:24] 
            car_head_y = attributes[:, 24:25] 
            car_head_w = attributes[:, 25:26] 
            car_head_h = attributes[:, 26:27] 
            car_plate_seen_conf = attributes[:, 27:28]
            car_plate_x = attributes[:, 28:29] 
            car_plate_y = attributes[:, 29:30] 
            car_plate_w = attributes[:, 30:31] 
            car_plate_h = attributes[:, 31:32] 
            person_type_conf, person_type = attributes[:, 32:34].max(1, keepdims=True)
            person_status_conf, person_status_type = attributes[:, 34:36].max(1, keepdims=True)
            if_helmet_on_conf, if_helmet_on = attributes[:, 36:38].max(1, keepdims=True)
            person_head_seen_conf = attributes[:, 38:39]
            person_head_x = attributes[:, 39:40] 
            person_head_y = attributes[:, 40:41] 
            person_head_w = attributes[:, 41:42] 
            person_head_h = attributes[:, 42:43] 
            person_bike_seen_conf = attributes[:, 43:44]
            person_bike_x = attributes[:, 44:45] 
            person_bike_y = attributes[:, 45:46] 
            person_bike_w = attributes[:, 46:47] 
            person_bike_h = attributes[:, 47:48] 
            bike_plate_seen_conf = attributes[:, 48:49]
            bike_plate_x = attributes[:, 49:50] 
            bike_plate_y = attributes[:, 50:51] 
            bike_plate_w = attributes[:, 51:52] 
            bike_plate_h = attributes[:, 52:53] 
            traffic_light_color_conf, traffic_light_color = attributes[:, 53:56].max(1, keepdims=True)
            road_sign_color_conf, road_sign_color = attributes[:, 56:58].max(1, keepdims=True)
            # others
            others = attributes[:, 58:]
             
            # 其中 -1 是为了占位
            attributes = torch.cat((
                car_type, car_type_conf, place_holder, 
                car_left_light_seen_conf, car_left_light_status, car_left_light_seen_conf, car_left_light_x, car_left_light_y, car_left_light_w, car_left_light_h,
                car_right_light_seen_conf, car_right_light_status, car_right_light_status_conf, car_right_light_x, car_right_light_y, car_right_light_w, car_right_light_h,
                car_butt_seen_conf, car_butt_x, car_butt_y, car_butt_w, car_butt_h, 
                car_head_seen_conf, car_head_x, car_head_y, car_head_w, car_head_h,
                car_plate_seen_conf, car_plate_x, car_plate_y, car_plate_w, car_plate_h,
                person_type, person_status_conf, person_status_type, person_status_conf, if_helmet_on, if_helmet_on_conf,
                person_head_seen_conf, person_head_x, person_head_y, person_head_w, person_head_h,
                person_bike_seen_conf, person_bike_x, person_bike_y, person_bike_w, person_bike_h,
                bike_plate_seen_conf, bike_plate_x, bike_plate_y, bike_plate_w, bike_plate_h,
                traffic_light_color, traffic_light_color_conf, place_holder, 
                road_sign_color, road_sign_color_conf,
                others
            ), 1)

        # Detections matrix nx6 (xyxy, conf, cls, attributes)
        if multi_label:
            i, j = (x[:, 5 : nc] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), attributes[i]), 1)
        else:  # best class only
            conf, class_index = x[:, 5 : 5 + nc].max(1, keepdim=True)
            x = torch.cat((box, conf, class_index.float(), attributes), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        # c 指的是是否对不同类别的 bounding box 进行偏置，从而在进行 nms 的时候不同类别的框不会重叠
        offset_by_classes = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + offset_by_classes, x[:, 4]  # boxes (offset by class), scores

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output






# 定义需要的参数
output_dir_bus = "/workspace/data/vehicles_croped_from_datasets/bus"  # bus 输出文件夹路径
output_dir_truck = "/workspace/data/vehicles_croped_from_datasets/truck"  # truck 输出文件夹路径
output_dir_others = "/workspace/data/vehicles_croped_from_datasets/others"  # truck 输出文件夹路径
vehicle_index = 0  # 在检测数据集中所有车辆的标签索引
model_path = "/workspace/home/chenyichen/yolov7/scripts/vehicleType_recog.onnx"  # ONNX模型文件路径

# 加载ONNX模型
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 3,
    }),
    'CPUExecutionProvider',
]
session = ort.InferenceSession(model_path, providers=providers)

if False:
    # 获取图片文件列表
    with open("./all_train_data.txt", "r") as f:
        img_files = f.read().splitlines()

    # 遍历每张图片
    num_bus_croped = 0
    num_truck_croped = 0
    num_others_croped = 0
    for num, img_file in enumerate(img_files):
        if num_bus_croped > 5000 and num_truck_croped > 5000:
            break

        img_path = img_file.strip()
        print('processing ' + img_path)
        print("num_bus_croped:" + str(num_bus_croped) + " num_truck_croped:" + str(num_truck_croped) + " num_others_croped:" + str(num_others_croped))

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 获取对应的标签文件
        label_file = img_path[:-4] + '.txt'
        if not os.path.exists(label_file):
            continue

        # 读取标签文件中的物体信息
        with open(label_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                class_id, center_x, center_y, bbox_w, bbox_h = map(float, line.strip().split())

                # 如果目标类别索引等于指定索引并且置信度超过阈值
                if class_id == vehicle_index:
                    # 计算物体的位置信息
                    x1 = int((center_x - bbox_w / 2) * w)
                    y1 = int((center_y - bbox_h / 2) * h)
                    x2 = int((center_x + bbox_w / 2) * w)
                    y2 = int((center_y + bbox_h / 2) * h)

                    # 面积小于 100x100 的丢掉
                    area = abs(x2 - x1) * abs(y2 - y1)
                    if area < 10000:
                        continue

                    # 截取小图块
                    crop = img.crop((x1, y1, x2, y2))

                    # 对小图块进行分类
                    input_data = np.asarray(crop.resize((224, 224)), dtype=np.float32).transpose((2, 0, 1)) / 255.0
                    input_data = np.expand_dims(input_data, axis=0)
                    output = session.run(None, {'input.1': input_data})[0]

                    # 保存截取下来的小图块到指定的输出文件夹中
                    class_index = np.argmax(output)

                    if class_index == 1:
                        if num_bus_croped > 5000:
                            continue
                        output_dir = output_dir_bus
                        num_bus_croped += 1
                    elif class_index == 2:
                        if num_truck_croped > 5000:
                            continue
                        output_dir = output_dir_truck
                        num_truck_croped += 1
                    elif class_index == 0:
                        if num_others_croped > 5000:
                            continue
                        output_dir = output_dir_others
                        num_others_croped += 1

                    crop.save(os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + '_crop_' + str(i) + '.jpg'))

if True:
    # 获取视频文件夹
    videos = [video for video in os.listdir("/workspace/data/vehicles_croped_from_datasets/") if video.endswith(".mp4")]

    detector_model_path = "/workspace/home/chenyichen/yolov7/chezai_box_class12/yolov7-tiny/weights/best_withoutNms.onnx"
    # 加载ONNX模型
    detector_model_providers = [
        ('CUDAExecutionProvider', {
            'device_id': 2,
        }),
        'CPUExecutionProvider',
    ]
    detector_model_session = ort.InferenceSession(detector_model_path, providers=detector_model_providers)


    # 遍历每个视频
    num_bus_croped = 0
    num_truck_croped = 0
    num_others_croped = 0
    for video in videos:
        # 打开视频
        video_file = cv2.VideoCapture("/workspace/data/vehicles_croped_from_datasets/" + video)

        print("num_bus_croped:" + str(num_bus_croped) + " num_truck_croped:" + str(num_truck_croped) + " num_others_croped:" + str(num_others_croped))
        print('processing ' + video)

        # 获取视频的帧率
        framerate = video_file.get(cv2.CAP_PROP_FPS)

        # 遍历视频的每一帧
        i = 0
        while True:

            # 读取下一帧
            ret, frame = video_file.read()

            # 如果没有下一帧，则退出循环
            if not ret:
                break

            # 将帧转换为numpy数组
            ori_img_w = frame.shape[1]
            ori_img_h = frame.shape[0]
            ori_img = frame
            frame = cv2.resize(frame, (640, 640))
            img = frame
            frame = np.array(frame).astype(np.float32) / 255.0     
            frame = frame[np.newaxis, :].transpose(0, 3, 1,2)

            # 每秒抽取5帧
            i += 1
            if i % 5 == 0:
                # 对帧进行推理
                output = detector_model_session.run(None, {"images": frame})
                out_tensor = torch.from_numpy(output[0])
                out = non_max_suppression(out_tensor, conf_thres=0.4, iou_thres=0.3, multi_label=False, attribute_outputs=0)
                
                for o in out[0]:
                    o = o.numpy()
                    x1 = o[0] * ori_img_w / 640
                    y1 = o[1] * ori_img_h / 640
                    x2 = o[2] * ori_img_w / 640
                    y2 = o[3] * ori_img_h / 640
                    p = o[4]
                    c = o[5]
                    
                    if(c == 0):
                        crop = ori_img.copy()[int(y1):int(y2), int(x1):int(x2)].copy()
                        if(crop.size != 0):

                            # 对小图块进行分类
                            crop_resized = cv2.resize(crop, (256, 256))
                            input_data = np.asarray(crop_resized, dtype=np.float32).transpose((2, 0, 1)) / 255.0
                            input_data = np.expand_dims(input_data, axis=0)
                            classify_output = session.run(None, {'input': input_data})[0]

                            # 保存截取下来的小图块到指定的输出文件夹中
                            class_index = np.argmax(classify_output)

                            if(class_index == 0):
                                cv2.imwrite("./output/bus/" + video[:-4] + "_" + str(i) + ".png", crop)
                            elif(class_index == 1):
                                cv2.imwrite("./output/car/" + video[:-4] + "_" + str(i) + ".png", crop)
                            else:
                                cv2.imwrite("./output/truck/" + video[:-4] + "_" + str(i) + ".png", crop)


                        #cv2.rectangle(ori_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))


        # 关闭视频
        video_file.release()