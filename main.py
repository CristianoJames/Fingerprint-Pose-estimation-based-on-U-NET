import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import UNet

from torch.autograd import Variable
import matplotlib.pyplot as plt
# 是否使用cuda


import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms

from dataset import fingerprintDataset
from mIou import *
import os
import cv2
# 是否使用cuda
import PIL.Image as Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(i):
    import dataset
    imgs = dataset.make_dataset(r"E:\360Downloads\dataset\fingerprint\val")
    imgx = []
    imgy = []
    for img in imgs:
        imgx.append(img[0])
        imgy.append(img[1])
    return imgx[i],imgy[i]

x_transforms = transforms.Compose([

    #transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),

])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
   # transforms.Resize((512, 512)),
    transforms.ToTensor(),


])







def train_model(model, criterion, optimizer, dataload, num_epochs=500):
    Loss_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0

        for x, y in dataload:
            step += 1



            # zero the parameter gradients




            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #accu = dice_coeff(outputs, labels)
            loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.5f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
            #print(accu)
        print("epoch %d loss:%0.5f" % (epoch, epoch_loss/step))
        Loss_list.append(epoch_loss / step)


    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    x=range(0,500)
    y= Loss_list
    plt.plot(x, y, '.-')
    plt.ylabel('Loss')
    plt.xlabel('x')
    plt.show()
    return model


#训练模型


def train(args):
    model = UNet(1).to(device)

    batch_size = args.batch_size

    LR = 0.005
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCEWithLogitsLoss()

    lr_list = []
    liver_dataset =LiverDataset(r"E:\360Downloads\dataset\fingerprint\train3",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory= True)
    train_model(model, criterion, optimizer, dataloaders)
def test(args):

#显示模型的输出结果
    model = UNet(1).to(device)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))

    liver_dataset = LiverDataset(r"E:\360Downloads\dataset\fingerprint\val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()






    import matplotlib.pyplot as plt

    plt.ion()  # 开启动态模式

    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        num = len(dataloaders)  # 验证集图片的总数
        for x, _ in dataloaders:
            x = x.to(device)
            y = model(x)

            img_y = torch.squeeze(y).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            mask = get_data(i)[1]  # 得到当前mask的路径
            miou_total += get_iou(mask, img_y)  # 获取当前预测图的miou，并加到总miou中
            plt.subplot(1,3,1)
            plt.imshow(Image.open(get_data(i)[0]))
            plt.title('Input fingerprint image')
            plt.subplot(1,3,2)
            plt.imshow(Image.open(get_data(i)[1]))
            plt.title('Ground Truth label')
            plt.subplot(1,3,3)
            plt.imshow(img_y)
            plt.title('Estimated fingerprint pose')
            plt.pause(20)
            if i < num: i += 1  # 处理验证集下一张图
        plt.show()
        print('Miou=%f' % (miou_total / 100))




if __name__ == '__main__':
   
    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    parse.add_argument( '--learning-rate', dest='lr', type=float, default=0.01, help='learning rate')
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action == "test":
        test(args)
