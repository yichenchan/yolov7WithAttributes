import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


# # 数据集划分
# import os
# from shutil import copy
# import random

# def mkfile(file):k
#     if not os.path.exists(file):
#         os.makedirs(file)
 
# # 获取data文件夹下所有文件夹名（即需要分类的类名）
# file_path = '/workspace/data/vehicles_croped_from_datasets/all'
# car_class = [cla for cla in os.listdir(file_path)]
 
# # 创建 训练集train 文件夹，并由类名在其目录下创建5个子目录
# mkfile('/workspace/data/vehicles_croped_from_datasets/train')
# for cla in car_class:
#     mkfile('/workspace/data/vehicles_croped_from_datasets/train/' + cla)
 
# # 创建 验证集val 文件夹，并由类名在其目录下创建子目录
# mkfile('/workspace/data/vehicles_croped_from_datasets/val')
# for cla in car_class:
#     mkfile('/workspace/data/vehicles_croped_from_datasets/val/' + cla)
 
# # 划分比例，训练集 : 验证集 = 9 : 1
# split_rate = 0.1
 
# # 遍历所有类别的全部图像并按比例分成训练集和验证集
# for cla in car_class:
#     cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
#     images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
#     num = len(images)
#     eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
#     for index, image in enumerate(images):
#         # eval_index 中保存验证集val的图像名称
#         if image in eval_index:
#             image_path = cla_path + image
#             new_path = '/workspace/data/vehicles_croped_from_datasets/val/' + cla
#             copy(image_path, new_path)  # 将选中的图像复制到新路径
 
#         # 其余的图像保存在训练集train中
#         else:
#             image_path = cla_path + image
#             new_path = '/workspace/data/vehicles_croped_from_datasets/train/' + cla
#             copy(image_path, new_path)
#         print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
#     print(cla)
 
# print("processing done!")

# 基础网络
@model_wrapper      # this decorator should be put on the out most
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(774400, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output

# 定义网络搜索空间
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding='same')
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding='same')

    def forward(self, x):
        return F.relu(self.pointwise(self.depthwise(x)))


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding='same')
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        # 在普通的 conv 层和定制的 DepthwiseSeparableConv 层之间选择
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')

        self.block3 = nn.Repeat(DepthwiseSeparableConv(64, 64), nn.ValueChoice([1, 2, 3]))

        self.conv4 = nn.Conv2d(64, 128, 3, 1, padding='same')

        self.block5 = nn.Repeat(DepthwiseSeparableConv(128, 128), nn.ValueChoice([1, 2]))

        # ValueChoice is used to select a dropout rate.
        # ValueChoice can be used as parameter of modules wrapped in `nni.retiarii.nn.pytorch`
        # or customized modules wrapped with `@basic_unit`.
        # 在不同的 dropout 的率之间选择
        self.dropout1 = nn.Dropout(0.75)  # choose dropout rate from 0.25, 0.5 and 0.75
        self.dropout2 = nn.Dropout(0.5)
        #feature = nn.ValueChoice([64, 128, 256, 512])
        self.fc1 = nn.Linear(401408, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x) # 32 x 224 x 224
        x = F.relu(x)
        x = self.conv2(x) # 64 x 224 x 224 
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 64 x 112 x 112 
        x = x + self.block3(x) # 64 x 112 x 112 
        x = self.conv4(x) # 128 x 112 x 112
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 128 x 56 x 56       
        x = x + self.block5(x) # 128 x 56 x 56
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 401408 x 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

model_space = ModelSpace()
print(model_space)

# 定义搜索策略
import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted

# 定义评价指标
import nni

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms


def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 裁剪为224*224
        transforms.RandomHorizontalFlip(), # 随机垂直旋转
        transforms.ColorJitter(brightness=1),# 亮度随机变化
        transforms.ColorJitter(contrast=1), # 对比度随机变化
        transforms.ColorJitter(hue=0.5), # 颜色随机变化
        transforms.ToTensor(), # 将0-255范围内的像素转为0-1范围内的tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_dataset = ImageFolder('/workspace/data/vehicles_croped_from_datasets/train/', transform=train_transform)
    val_dataset = ImageFolder('/workspace/data/vehicles_croped_from_datasets/val/', transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    for epoch in range(20):
        # train the model for one epoch
        train_epoch(model, device, train_dataloader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, val_dataloader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)


import os
from pathlib import Path

def evaluate_model_with_visualization(model_cls):
    model = model_cls()
    # dump the model into an onnx
    if 'NNI_OUTPUT_DIR' in os.environ:
        dummy_input = torch.zeros(1, 3, 32, 32)
        torch.onnx.export(model, (dummy_input, ),
                          Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    evaluate_model(model_cls)

# 创造评价器
from nni.retiarii.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model)

# 开始搜索
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'vehicle_classification_search'
exp_config.experiment_working_directory = "./vehicleType_classification_nas_experiments"

exp_config.max_trial_number = 20  # spawn 4 trials at most
exp_config.trial_concurrency = 4  # will run two trials concurrently

exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True

exp.run(exp_config, 7000)

for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)