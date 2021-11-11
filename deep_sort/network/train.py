import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from  torch.utils.data import Dataset, DataLoader
import argparse
import time
import argparse
import warnings

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model import CNN

params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 6}

def make_parser():
    parser = argparse.ArgumentParser("DS Network train parser")
    parser.add_argument("--path",default=None,type=str, help=" data path for training")
    
    parser.add_argument("--lr",default=0.1, type=float, help=" learning rate ")
    parser.add_argument('--resume', '-r',action='store_true', help=" resume training ")
    # parser.add_argument("--gpu-id",default=0,type=int)
    # parser.add_argument("--interval",'-i',default=20,type=int)

    parser.add_argument("--epoch",'-e', default=3, type=int, help=" number of epoch ")
    # parser.add_argument("--output",'-e', default=10, type=int, help=" output model ")
    # parser.add_argument("--epoch",'-e', default=10, type=int, help=" number of epoch ")
    args = parser.parse_args()
    return args


def check_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        # Enables benchmark mode in cudnn
        cudnn.benchmark = True
    return device

def data_loader(args):
    train_dir = os.path.join(args.path,"train")
    val_dir = os.path.join(args.path,"val")

    training_init  = transforms.Compose([
        transforms.Resize((128,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainloader = DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=training_init),
        **params
    )

    validation_init = transforms.Compose([
        transforms.Resize((128,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    valloader = DataLoader(
        torchvision.datasets.ImageFolder(val_dir, transform=validation_init),
        **params
    )
    # print(f"Num Class {len(trainloader.dataset.classes)} - {len(valloader.dataset.classes)}")
    num_classes = max(len(trainloader.dataset.classes), 
        len(valloader.dataset.classes))

    return num_classes, trainloader, valloader

def train(epoch):
    print(f"\nEpoch : {epoch+1}")
    model.train()
    sum_train_loss = 0.
    correct = 0
    total = 0

    start = time.time()
    for counter, (images, labels) in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        images ,labels = images.to(device),labels.to(device)
        # print(f"Image Shape {images.shape}") #torch.Size([32, 3, 128, 64]) 32 is a number of batch

        # zero the param gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # accumulating train process
        sum_train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # print statistics
        if (counter+1)%100 == 0: #print every 10 input
            end = time.time()
            print(f"[Current:{counter}/{len(trainloader)}], time:{(end-start):.2f}s Acc:{100.*correct/total:.3f}%")
            start = time.time()

    sum_train_loss=sum_train_loss/len(trainloader)
    train_err=1.-correct/total
    acc = 100.*correct/total
    
    print(f'\nEpoch {epoch+1} Train Loss: {sum_train_loss:.3f}% | Train Error: {train_err:.3f} |Accuracy: {acc:.3f}%')

    return sum_train_loss, train_err, acc

# lr decay
def lr_decay(optimizer):
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print(f"\nLearning rate adjusted to {lr}")    

def save_model(checkpoint, acc, save_dir='checkpoint'):
    #Save ckpt, evaluate
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "ckpt.pth")
    torch.save(checkpoint, filename)
    print(f"Checkpoint Saved on {filename}")

# plot figure
def draw_curve(epoch, train_loss, train_err): #test_loss, test_err
    x_epoch = []
    record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    # record['test_loss'].append(test_loss)
    # record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    # ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    # ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("graph.jpg")


if __name__ == '__main__':
    args = make_parser()
    device = check_device()
    num_classes, trainloader, valloader = data_loader(args)
    # print(num_classes)

    # initiate_model, criterion, optimizer, res_epoch = before_train(args, num_classes, device)
    #Network definition 
    model = CNN(num_classes=num_classes)
    
    if args.resume:
        assert os.path.isfile("./checkpoint/ckpt.pth"), "Error: no checkpoint file found!"
        print('Loading from checkpoint/ckpt.pth')
        checkpoint = torch.load("./checkpoint/ckpt.pth")

        print(f"\n Last Epoch {checkpoint['epoch']} | Accuracy: {checkpoint['acc']:.3f}%")
        
        net_dict = checkpoint['net_dict']
        model.load_state_dict(net_dict)
        # best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] # Need to be tested
    else:
        start_epoch = 0 # Need to be tested
    model.to(device)

    #Loss and optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9) #weight_decay=5e-4
    #best_acc = 0.

    train_loss_ar=[]
    train_er_ar=[]
    acc_ar=[]
    best_acc=0.0

    for epoch in range(start_epoch, start_epoch+args.epoch):
        train_loss, train_err, acc = train(epoch)
        train_loss_ar.append(train_loss)
        train_er_ar.append(train_err)
        acc_ar.append(acc)
        # draw_curve(epoch, train_loss, train_err)
        
        if (epoch+1)%20==0: #Not sure using 20, but less than 20, still learning significantly
            lr_decay(optimizer)

        if acc > best_acc:
            best_acc = acc
            checkpoint = {
                'net_dict':model.state_dict(),
                'acc':best_acc,
                'epoch':epoch,
            }
            save_model(checkpoint, best_acc)
    
    

