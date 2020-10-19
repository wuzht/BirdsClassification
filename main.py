import os
import argparse
import time
import datetime
import torch
import torchvision
import torch.nn as nn

from dataset import CUB200Dataset
from tools import choose_gpu, Logger, AverageMeter, setup_seed, mkdir, summary
from tools import to_str_args, to_str_lr_scheduler, ImProgressBar

parser = argparse.ArgumentParser(description='CUB200 Classification  ')
parser.add_argument("--root", type=str, default='./CUB-200', help='data root dir')
parser.add_argument("--exp_dir", type=str, default='./exp', help='experiments dir')

def get_args():
    args = parser.parse_args()
    gpu_id, memory_gpu = choose_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    args.gpu_id = gpu_id
    args.memory_gpu = memory_gpu
    args.optim_type    = 'Adam'         # optimizer type, choices=['SGD', 'Adam']
    args.num_classes   = 200
    args.lr            = 1e-2
    args.batch_size    = 256
    args.momentum      = 0.9
    args.decay         = 1e-3
    args.epochs        = 300
    args.num_workers = 8
    args.pin_memory = True
    # args.seed = time.time()
    args.seed = 7
    setup_seed(args.seed)
    return args

class Trainer():
    def __init__(self, args):
        now_time = datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M%S')
        args.cur_dir = os.path.join(args.exp_dir, now_time)
        args.log_path = os.path.join(args.cur_dir, 'train.log')
        args.best_model_path = os.path.join(args.cur_dir, 'best_model.pth')

        self.args = args
        mkdir(self.args.exp_dir)
        mkdir(self.args.cur_dir)
        self.log = Logger(self.args.log_path, level='debug').logger
        self.log.critical("args: \n{}".format(to_str_args(self.args)))

        self.train_loader = torch.utils.data.DataLoader(
            dataset=CUB200Dataset(root=self.args.root, train=True),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=CUB200Dataset(root=self.args.root, train=False),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=False
        )

        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.args.num_classes)
        self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.args.lr
        ) if self.args.optim_type == 'Adam' else torch.optim.SGD(
            params=self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.decay
        )

        self.log.critical("model: \n{}".format(self.model))
        self.log.critical("torchsummary: \n{}".format(summary(model=self.model, input_size=(3, 224, 224))))
        self.log.critical("criterion: \n{}".format(self.criterion))
        self.log.critical("optimizer: \n{}".format(self.optimizer))

    def train(self):
        self.model.train()
        losses = AverageMeter()
        correct = 0
        pbar = ImProgressBar(len(self.train_loader))
        for i, (imgs, targets) in enumerate(self.train_loader):
            imgs, targets = imgs.cuda(), targets.cuda()
            outputs = self.model(imgs)
            
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == targets).sum().item()

            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), 1)
            pbar.update(i)
        pbar.finish()
        return losses.avg, correct / len(self.train_loader.dataset)

    def eval(self, loader):
        self.model.eval()
        losses = AverageMeter()
        correct = 0
        with torch.no_grad():
            pbar = ImProgressBar(len(loader))
            for i, (imgs, targets) in enumerate(loader):
                imgs, targets = imgs.cuda(), targets.cuda()
                outputs = self.model(imgs)

                _, predicted = torch.max(outputs.data, dim=1)
                correct += (predicted == targets).sum().item()

                loss = self.criterion(outputs, targets)
                losses.update(loss.item(), 1)
                
                pbar.update(i)
            pbar.finish()
        return losses.avg, correct / len(loader.dataset)

    def fit(self):
        best_epoch, best_test_acc = 0, 0
        for epoch in range(0, self.args.epochs):
            end = time.time()
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.eval(self.test_loader)
            
            if test_acc > best_test_acc:
                best_epoch = epoch
                best_test_acc = test_acc
                checkpoint = {
                    'epoch': epoch + 1,
                    'args': vars(self.args),
                    'state_dict': self.model.state_dict(),
                    'best_test_acc': best_test_acc,
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(checkpoint, self.args.best_model_path)

            self.log.info('[Epoch: {:3}/{:3}][Time: {:.3f}] Train loss: {:.3f}, Test loss: {:.3f}, Train acc: {:.3f}%, Test acc: {:.3f}% (best_test_acc: {:.3f}%, epoch: {})'.format(
                epoch + 1, self.args.epochs, time.time() - end, train_loss, test_loss, train_acc * 100, test_acc * 100, best_test_acc * 100, best_epoch))

        
if __name__ == "__main__":
    trainer = Trainer(get_args())
    trainer.fit()
    # train_loss, train_acc = trainer.eval(trainer.train_loader)
    # test_loss, test_acc = trainer.eval(trainer.test_loader)
    # trainer.log.info('Train loss: {:.3f}, Test loss: {:.3f}, Train acc: {:.3f}%, Test acc: {:.3f}%'.format(
    #     train_loss, test_loss, train_acc * 100, test_acc * 100))
