from PIL import Image
import os
import onnxruntime as ort
import numpy as np

# 定义需要的参数
output_dir_bus = "/workspace/data/vehicles_croped_from_datasets/bus"  # bus 输出文件夹路径
output_dir_truck = "/workspace/data/vehicles_croped_from_datasets/truck"  # truck 输出文件夹路径
output_dir_others = "/workspace/data/vehicles_croped_from_datasets/others"  # truck 输出文件夹路径
vehicle_index = 0  # 在检测数据集中所有车辆的标签索引
model_path = "/workspace/home/chenyichen/yolov7/scripts/vehicleType_recog.onnx"  # ONNX模型文件路径

# 加载ONNX模型
providers = ['CUDAExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

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
