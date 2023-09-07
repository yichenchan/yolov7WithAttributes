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
import onnx
import onnxruntime
import numpy
import cv2


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

    print('Accuracy: {}/{} ({:.2f}%)'.format(
          correct, len(val_loader.dataset), accuracy))

    return test_loss, accuracy


def test(args):
    model = torch.load(args.weight)['model']
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        ])

    test_dataset = ImageFolder(args.test_dataset_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8)
    val(model, 0, test_loader)


def onnx_test(args):
    ort_session = onnxruntime.InferenceSession(args.weight, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        ])
    test_dataset = ImageFolder(args.test_dataset_folder, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)

    total_correct = 0
    for data, target in test_dataloader:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
        ort_outs = ort_session.run(None, ort_inputs)
        pred = numpy.argmax(ort_outs[0], axis=1)
        if(int(pred[0]) == int(target[0])):
            total_correct += 1

    acc = total_correct / len(test_dataloader)
    print("Onnx Test Accuracy:", acc)

def export(args):
    model = torch.load(args.weight)['model'].to('cpu')
    img = torch.rand([1, 3, args.img_size, args.img_size])
    torch.onnx.export(
        model, 
        img,  
        args.models_folder + "model.onnx", 
        verbose=False, 
        opset_version=12, 
        input_names=['input'],
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

    if(args.resume):
        latest_model = max([os.path.join(args.models_folder, f) for f in os.listdir(args.models_folder)], key=os.path.getmtime)
        model = torch.load(latest_model)['model'].type(torch.cuda.FloatTensor)
        trained_epochs = torch.load(latest_model)['epoch']
        optimizer_dict = torch.load(latest_model)['optimizer_dict']
    else:
        if(args.weight == ""):
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        else:
            print("loading weight", args.weight)
            model = torch.load(args.weight)['model'].type(torch.cuda.FloatTensor)
            #model = torch.load(args.weight).type(torch.cuda.FloatTensor)
            #model.fc = nn.Linear(model.fc.in_features, args.num_classes)
            print("model loading done!")

    model.cuda(gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr_start)
    if(args.resume):
        optimizer.load_state_dict(optimizer_dict)
    #lr_lambda = lambda epoch: args.lr_start - ((args.lr_start - args.lr_final) * (epoch / (args.epochs - 1)))
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data loading code
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=0.25),# 亮度随机变化
        transforms.ColorJitter(contrast=0.25), # 对比度随机变化
        transforms.ColorJitter(hue=0.25), # 颜色随机变化
        transforms.ToTensor(), # 将0-255范围内的像素转为0-1范围内的tensor
        ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=0.25),# 亮度随机变化
        transforms.ColorJitter(contrast=0.25), # 对比度随机变化
        transforms.ColorJitter(hue=0.25), # 颜色随机变化
        transforms.ToTensor(),
        ])

    train_dataset = ImageFolder(args.train_dataset_folder, transform=train_transform)
    val_dataset = ImageFolder(args.val_dataset_folder, transform=val_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.single_gpu_training_batch_size,
        shuffle=False,            
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler)    

    if(gpu == 0 and args.dataloader_visualization):
        images, _ = next(iter(train_loader))
        for index, img in enumerate(images):
            img = img.numpy().transpose((1, 2, 0)) * 255
            cv2.imwrite(args.models_folder + '/' + str(index) + '.jpg', img)
        print("dataloader visualization done!")

    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=args.testing_batch_size, 
        shuffle=True, 
        num_workers=args.num_workers)

    if(gpu == 0):
        wandb.login()
        run = wandb.init(project=args.project, name=args.exp_name, config=args)

    start = datetime.now()
    total_step = len(train_loader)

    for epoch in range(args.epochs):
        # 跳过已经训练过的 epochs
        if(args.resume and epoch < trained_epochs):
            continue

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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Lr: {:.8f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    train_loss.item(),
                    optimizer.state_dict()['param_groups'][0]['lr']))

        if(gpu == 0):
            print("train dataset testing...")
            train_loss, train_acc = val(model, gpu, train_loader)
            print("val dataset testing...")
            val_loss, val_acc = val(model, gpu, val_loader)

            wandb.log({
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': val_loss
            })

            if(epoch % 5 == 0):
                ckpt = {'epoch': epoch,
                        'model': deepcopy(model.module),
                        'optimizer_dict': optimizer.state_dict()
                        }
            
                torch.save(ckpt, args.models_folder + '/epoch_{:03d}.pt'.format(epoch))

    if(gpu == 0):
        print("Training complete in: " + str(datetime.now() - start))

def main():
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
    parser.add_argument('-nc', '--num_classes', default=3, type=int)
    parser.add_argument('-imgsz', '--img_size', default=256, type=int)
    parser.add_argument('--testing_batch_size', default=128, type=int)
    parser.add_argument('-nw', '--num_workers', default=8, type=int)
    parser.add_argument('--project', default='new project', type=str)
    parser.add_argument('--exp_name', default='new exp', type=str)
    parser.add_argument('--train_dataset_folder', default='', type=str)
    parser.add_argument('--val_dataset_folder', default='', type=str)
    parser.add_argument('--test_dataset_folder', default='', type=str)
    parser.add_argument('--weight', default='', type=str)
    parser.add_argument('--task', default='train', type=str)
    parser.add_argument('--models_folder', default='./vehicleTypeModels/', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dataloader_visualization', action='store_true')
    args = parser.parse_args()

    if(args.task == 'train'):
        #########################################################
        args.world_size = args.gpus * args.nodes                # 总进程数 = 每台服务器 gpu 数量 x 总服务器数
        os.environ['MASTER_ADDR'] = '0.0.0.0'                   # 主节点的 ip 地址
        os.environ['MASTER_PORT'] = '7000'                      # 主节点的端口号
        mp.spawn(train, nprocs=args.gpus, args=(args,))         # 开启 args.gpus 个数目的 train(i, args)进程，i 是 0～args.gpus
        ######################################################### 
    elif(args.task == 'test'):
        test(args)
    elif(args.task == 'export'):
        export(args)
    elif(args.task == 'onnx_test'):
        onnx_test(args)

if __name__ == '__main__':
    main()
