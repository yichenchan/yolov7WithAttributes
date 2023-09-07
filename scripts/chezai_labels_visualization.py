import os
import cv2

def put_on_text(image, text, x, y, color):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, y - text_height - baseline - 5), (x + text_width + 5, y), color, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return x, y - text_height - baseline - 5

def main():
    # Set the path to the dataset file
    dataset_file = './chezai_dataset_with_attr.txt'

    # Set the class names
    class_names = [
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

    # Read the dataset file
    with open(dataset_file, 'r') as f:
        label_files = [line.strip() for line in f]

    for label_file in label_files:
        print("deal with ", label_file)

        # Read the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Load the corresponding image
        image_file = os.path.splitext(label_file)[0] + '.jpg'
        image_file_osd = os.path.splitext(label_file)[0] + '_labeled.jpg'
        image = cv2.imread(image_file)
        img_h, img_w = image.shape[:2]
        # Parse the bounding boxes
        for line in lines:
            items = line.strip().split()
            if len(items) >= 5:
                c = int(items[0])
                x = float(items[1])
                y = float(items[2])
                w = float(items[3])
                h = float(items[4])
                x1 = int((x - w / 2) * img_w)
                y1 = int((y - h / 2) * img_h)
                x2 = int((x + w / 2) * img_w)
                y2 = int((y + h / 2) * img_h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = class_names[c]
                new_x, new_y = put_on_text(image, label, x1, y1, (0, 0, 255))

                if c == 0:
                    car_type = "car_type:" + items[5]
                    new_x, new_y = put_on_text(image, car_type, new_x, new_y, (0, 255, 0))
                    




        # Display the image
        cv2.imwrite(image_file_osd, image)

if __name__ == '__main__':
    main()


# 车辆 32 + 人 21 + 红绿灯路牌 5 + 四种线 32 = 90
# 属性顺序 [车辆类别（3），左车灯是否可见（1），左车灯状态（2），左车灯位置（4），
#                      右车灯是否可见（1），右车灯状态（2），右车灯位置（4），
#                      车屁股是否可见（1），车屁股框位置（4）
#                      车头是否可见（1），车头框位置（4）
#                      车牌是否可见（1），车牌位置（4），
#         人的类别（2），人的状态（2）， 是否带头盔（2）
#                     人头是否可见（1），人头的位置（4），
#                     人骑的车是否可见（1），人骑的车的位置（4）
#                     电动车车牌是否可见（1），车牌的位置（4），
#         红绿灯的颜色（3），
#         路牌的颜色（2），
#         公交车道第一个点坐标（2），公交车道第二个点坐标（2），公交车道第三个点坐标（2），公交车道第四个点坐标（2），
#         斑马线第一个点坐标（2），斑马线第二个点坐标（2），斑马线第三个点坐标（2），斑马线第四个点坐标（2），
#         网格线第一个点坐标（2），网格线第二个点坐标（2），网格线第三个点坐标（2），网格线第四个点坐标（2），
#         导流线第一个点坐标（2），导流线第二个点坐标（2），导流线第三个点坐标（2），导流线第四个点坐标（2）]