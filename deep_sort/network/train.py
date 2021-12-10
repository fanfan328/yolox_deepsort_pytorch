import os
import os.path as osp
import time
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from model import CNN
from model_2 import Net

# reid import
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, write_mat_csv


def get_params():
    return args, params


def check_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        # Enables benchmark mode in cudnn
        cudnn.benchmark = True
    return device


# def load_model():
#     assert os.path.isfile("checkpoint/labeled_pedestrian.t7"), "Error: no checkpoint file found!"
#     print('Loading from checkpoint/ckpt.t7')
#     checkpoint = torch.load(f"./checkpoint/{args.checkpoint}.t7")
#     print(f"\n Last Epoch {checkpoint['epoch']} | Accuracy: {checkpoint['acc']:.3f}%")
#     return checkpoint


def get_data_loader(args):
    if args.dataset == 'viprior':
        return viprior_loader(args)
    elif args.dataset == 'spacejam' or 'pep':
        num_classes, train_loader, val_loader = data_loader(args)
        return '_', num_classes, train_loader, val_loader, '_'


def viprior_loader(args):
    root = args.path
    name = 'synergyreid'

    dataset = datasets.create(name, root, split_id=args.split)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if args.combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if args.combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=args.batch, num_workers=params['num_workers'],
        shuffle=True, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(list(set(dataset.query_val) | set(dataset.gallery_val)),
                     root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=args.batch, num_workers=params['num_workers'],
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query_test) | set(dataset.gallery_test)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=args.batch, num_workers=params['num_workers'],
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def data_loader(args):
    train_dir = os.path.join(args.path, "train")
    val_dir = os.path.join(args.path, "val")

    training_init = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainloader = DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=training_init),
        **params
    )

    validation_init = transforms.Compose([
        transforms.Resize((128, 64)),
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


def train(epoch, model, optimizer, criterion, train_loader, device):
    print(f"\nEpoch : {epoch + 1}")
    model.train()
    sum_train_loss = 0.
    correct = 0
    total = 0

    start = time.time()
    for counter, (images, labels) in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = images.to(device), labels.to(device)
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
        if (counter + 1) % 100 == 0:  # print every 10 input
            end = time.time()
            print(
                f"[Current:{counter + 1}/{len(train_loader)}], time:{(end - start):.2f}s Acc:{100. * correct / total:.3f}%")
            start = time.time()

    sum_train_loss = sum_train_loss / len(train_loader)
    train_err = 1. - correct / total
    acc = 100. * correct / total

    print(f'\nEpoch {epoch + 1} Train Loss: {sum_train_loss:.3f}% | Train Error: {train_err:.3f} |Accuracy: {acc:.3f}%')

    return sum_train_loss, train_err, acc

# lr decay
def lr_decay(optimizer):
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print(f"\nLearning rate adjusted to {lr}")


def save_model(checkpoint, acc, save_dir='checkpoint'):
    # Save ckpt, evaluate
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "ckpt.t7")
    torch.save(checkpoint, filename)
    print(f"Checkpoint Saved on {filename}")


# plot figure
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):  # test_loss, test_err
    x_epoch = []
    record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("graph.jpg")


def main(args, params):
    writer = SummaryWriter()

    # Create Dataloader
    dataset, num_classes, train_loader, val_loader, test_loader = get_data_loader(args)
    # if not args.num_classes == -1:
    #     nc = args.num_classes
    # else:
    # nc = num_classes

    # Create Model
    if args.model == 1:
        model = CNN(num_classes=num_classes)
    else:
        model = Net(num_classes=num_classes)

    # Load Model
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(osp.join(args.logs_dir, f'model_{args.model}_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        best_top1 = checkpoint['best_top1']
        start_epoch = checkpoint['epoch']  # Need to be tested
    device = check_device()
    model.to(device)

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model, writer)
    if args.evaluate and args.dataset == 'viprior':
        metric.train(model, train_loader)
        print("Validation:")
        dist_matrix = evaluator.evaluate(val_loader, dataset.query_val,
                                         dataset.gallery_val, metric)
        top1 = evaluator.compute_score(dist_matrix,
                                       dataset.query_val,
                                       dataset.gallery_val)
        write_mat_csv(osp.join(args.logs_dir, 'distance_matrix_val.csv'),
                      dist_matrix, dataset.query_val, dataset.gallery_val)
        print("Validation Top1 : {}".format(top1))
        print("Test:")
        dist_matrix = evaluator.evaluate(test_loader, dataset.query_test,
                                         dataset.gallery_test, metric)
        write_mat_csv(osp.join(args.logs_dir, 'distance_matrix.csv'),
                      dist_matrix, dataset.query_test, dataset.gallery_test)

    # Loss and optimization
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9)  # weight_decay=5e-4
    trainer = Trainer(model, criterion, writer)

    # Schedule learning rate
    def adjust_lr(epoch):
        # step_size = 60 if args.arch == 'inception' else 40
        step_size = 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    for epoch in range(start_epoch, start_epoch + args.epoch):
        adjust_lr(epoch)
        # train_loss, train_err, acc = train(epoch)
        if args.dataset == 'viprior':
            trainer.train(epoch, train_loader, optimizer)

            dist_matrix = evaluator.evaluate(val_loader,
                                             dataset.query_val,
                                             dataset.gallery_val)
            top1 = evaluator.compute_score(dist_matrix,
                                           dataset.query_val,
                                           dataset.gallery_val,
                                           epoch)
            writer.add_scalar('Test/Top1_avg', top1, epoch)
        else:
            sum_train_loss, train_err, top1 = train(epoch, model, optimizer, criterion, train_loader, device)

        # if (epoch + 1) % 10 == 0:  # Not sure using 20, but less than 20, still learning significantly
        #     lr_decay(optimizer)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, model_name=args.model, fpath=osp.join(args.logs_dir, f'ckpt_model_{args.model}.pth.tar'))

    # Final test
    # print('Test with best model:')
    # checkpoint = load_checkpoint(osp.join(args.logs_dir, f'model_{args.model}_best.pth.tar'))
    # model.load_state_dict(checkpoint['state_dict'])
    # metric.train(model, train_loader)
    # dist_matrix = evaluator.evaluate(test_loader,
    #                                  dataset.query_test,
    #                                  dataset.gallery_test,
    #                                  metric)
    # write_mat_csv(osp.join(args.logs_dir, 'distance_matrix.csv'),
    #               dist_matrix, dataset.query_test, dataset.gallery_test)

    # if acc > best_acc:
    #     best_acc = acc
    #     checkpoint = {
    #         'net_dict': model.state_dict(),
    #         'acc': best_acc,
    #         'epoch': epoch,
    #     }
    #     save_model(checkpoint, best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DS Network train parser")
    parser.add_argument("--evaluate", action='store_true', help='Evaluation only')
    parser.add_argument("--dataset", type=str, help="Insert dataset name",
                        choices=['viprior', 'spacejam', 'pep'])
    parser.add_argument("--path", default=None, type=str, help=" data path for training")

    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--model", default=1, type=int, help="Network to use")
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='ckpt', help='checkpoint path')
    parser.add_argument('--resume', '-r', action='store_true', help=" resume training ")
    # parser.add_argument("--interval",'-i',default=20,type=int)

    parser.add_argument("--lr", default=0.1, type=float, help=" learning rate ")
    parser.add_argument('--batch', '-b', type=int, default=64, help='batch size count')
    parser.add_argument("--epoch", '-e', default=50, type=int, help=" number of epoch ")
    # parser.add_argument("--num_classes", '-nc', default=-1, type=int,
    #                     help="Num classes to load model from different dataset")
    # parser.add_argument("--output",'-e', default=10, type=int, help=" output model ")

    # viprior args
    parser.add_argument('--logs_dir', type=str, default='./logs')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--combine_trainval', action='store_true',
                        help="train and val sets together for trainig, "
                             "val set alone for validation")
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme', 'lsml'])
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=64)
    args = parser.parse_args()

    params = {
        'batch_size': args.batch,
        'shuffle': True,
        'num_workers': 8,
    }
    main(args, params)
