import torch
import torch.nn.functional as F
from torch import nn
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from copy import deepcopy
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import math


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding='same')
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding='same')

    def forward(self, x):
        return F.relu(self.pointwise(self.depthwise(x)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.block1 = DepthwiseSeparableConv(32, 32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.block2 = DepthwiseSeparableConv(64, 64)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        
        self.dropout1 = nn.Dropout(0.3)  
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(100352, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 16 x 112 x 112

        x = self.conv2(x)  
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 32 x 56 x 56

        x = self.block1(self.block1(x)) # 32 x 56 x 56

        x = self.conv3(x) 
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 64 x 28 x 28

        x = self.block2(x) # 64 x 28 x 28

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2) # 128 x 14 x 14
        x = self.dropout1(x)

        x = torch.flatten(x, 1) # 100352 x 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


def val(model, device, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss().cuda(device)
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)

    print('\nVal Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(val_loader.dataset), accuracy))

    return test_loss, accuracy

def test():
    model = torch.load('/workspace/home/chenyichen/yolov7/vehicleTypeModels/resnet18_prunned0.5_class3_vehicleClassifyPretrain.pt')['model']
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_dataset = ImageFolder('/workspace/data/vehicles_croped_from_datasets/val/', transform=val_transform)

    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=8)
    val(model, 0, val_dataloader)


import onnx
import onnxruntime
import numpy
def onnx_test():
    ort_session = onnxruntime.InferenceSession("/workspace/home/chenyichen/yolov7/vehicleTypeModels/resnet18_prunned0.5_class3_vehicleClassifyPretrain.onnx", providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_dataset = ImageFolder('/workspace/data/vehicles_croped_from_datasets/val/', transform=val_transform)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8)
    total_correct = 0
    for data, target in val_dataloader:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
        ort_outs = ort_session.run(None, ort_inputs)
        pred = numpy.argmax(ort_outs[0], axis=1)
        if(int(pred[0]) == int(target[0])):
            total_correct += 1
    acc = total_correct / len(val_dataloader)
    print(acc)



def export():
    model = torch.load('/workspace/home/chenyichen/yolov7/vehicleTypeModels/resnet18_prunned0.5_class3_vehicleClassifyPretrain.pt')['model'].to('cpu')
    img = torch.rand([1, 3, 224, 224])
    torch.onnx.export(model, img, '/workspace/home/chenyichen/yolov7/vehicleTypeModels/resnet18_prunned0.5_class3_vehicleClassifyPretrain.onnx', verbose=False, opset_version=12, input_names=['input'],
                          output_names=['output'])


def train(gpu, args):
    torch.cuda.set_device(gpu)
    rank = args.nr * args.gpus + gpu	# 计算当前进程信号 = 服务器序号 x gpu 数量 + 当前 gpu 序号
                                        # 如果只有一台服务器, args.nr == 0
    # 初始化分布式环境                          
    dist.init_process_group(                                   
    	backend='nccl',                 # 通信后端，nccl是nvidia官方多卡通信框架                                         
   		init_method='env://',           # 初始化方式，env 是环境变量初始化方式
    	world_size=args.world_size,     # 总进程数量     
    	rank=rank)                      # 当前进程序号                        
    
    torch.manual_seed(0)
    # model = Net()
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    # state_dict = ckpt['model'].float().state_dict()  # to FP32
    # model.load_state_dict(state_dict, strict=False)  # load

    model = torch.load('/workspace/home/chenyichen/yolov7/vehicleTypeModels/resnet18_outClass500_vehicleClassifyPretrain.pt')['model'].type(torch.cuda.FloatTensor)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.cuda(gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr_start)
    # scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, eta_min=args.lr_final)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Data loading code
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(), # 随机垂直旋转
        transforms.ColorJitter(brightness=1),# 亮度随机变化
        transforms.ColorJitter(contrast=1), # 对比度随机变化
        transforms.ColorJitter(hue=0.5), # 颜色随机变化
        transforms.ToTensor(), # 将0-255范围内的像素转为0-1范围内的tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    train_dataset = ImageFolder('/workspace/data/vehicles_croped_from_datasets/train/', 
        transform=train_transform)

    val_dataset = ImageFolder('/workspace/data/vehicles_croped_from_datasets/val/', 
        transform=val_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.single_gpu_training_batch_size,
        shuffle=False,            
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler)    

    val_dataloader = DataLoader(val_dataset, batch_size=args.testing_batch_size, shuffle=True, num_workers=args.num_workers)

    if(gpu == 0):

        wandb.login()
        run = wandb.init(project=args.project, name=args.exp_name, config=args)

    start = datetime.now()
    total_step = len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            train_loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            #scheduler.step()

            if (i % 1 == 0 and gpu == 0):
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    train_loss.item()))

        if(gpu == 0):
            val_loss, val_acc = val(model, gpu, val_dataloader)

            wandb.log({
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss
            })

            ckpt = {'epoch': epoch,
                    'model': deepcopy(model.module),
                    'optimizer': optimizer.state_dict()
                    }
            
            torch.save(ckpt, './vehicleTypeModels/epoch_{:03d}.pt'.format(epoch))



    if(gpu == 0):
        print("Training complete in: " + str(datetime.now() - start))

def main():
    onnx_test()
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N') # 服务器的数量
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node') # 每台服务器的 gpu 数量
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes') # 每台服务器的序号 
    parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-lrs', '--lr_start', default=1e-3, type=float)
    parser.add_argument('-lrf', '--lr_final', default=1e-4, type=float)
    parser.add_argument('-bs', '--single_gpu_training_batch_size', default=128, type=int)
    parser.add_argument('--testing_batch_size', default=128, type=int)
    parser.add_argument('-nw', '--num_workers', default=8, type=int)
    parser.add_argument('--project', default='new project', type=str)
    parser.add_argument('--exp_name', default='new exp', type=str)
    args = parser.parse_args()

    #########################################################
    args.world_size = args.gpus * args.nodes                # 总进程数 = 每台服务器 gpu 数量 x 总服务器数
    os.environ['MASTER_ADDR'] = '0.0.0.0'                   # 主节点的 ip 地址
    os.environ['MASTER_PORT'] = '7000'                      # 主节点的端口号
    mp.spawn(train, nprocs=args.gpus, args=(args,))         # 开启 args.gpus 个数目的 train(i, args)进程，i 是 0～args.gpus
    ######################################################### 

if __name__ == '__main__':
    main()
