# coding: utf-8
import sys
import os
import re
import random
import itertools

import logging
from logging import handlers

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from collections import OrderedDict


def choose_gpu(gpu_not_use=[]):
    """
    return the id of the gpu with the most memory
    """
    # query GPU memory and save the result in `tmp`
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    # read the file `tmp` to get a gpu memory list
    memory_gpu = [int(x.split()[2]) for x in open('tmp','r').readlines()]

    for i in gpu_not_use:
        memory_gpu[i] = 0   # not use these gpus

    # get the id of the gpu with the most memory
    gpu_id = str(np.argmax(memory_gpu))
    # remove the file `tmp`
    os.system('rm tmp')

    # msg = 'memory_gpu: {}'.format(memory_gpu)
    return gpu_id, memory_gpu


def print_evironment():
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.current_stream())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=0, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式

        # https://docs.python.org/3/library/logging.handlers.html#timedrotatingfilehandler
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, interval=100, backupCount=backCount, encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除（默认为0，不删除），when是间隔的时间单位，单位有以下几种：
        # S 秒、M 分、H 小时、D 天、W 每星期（interval==0时代表星期一）、midnight 每天凌晨
        # 间隔 when * interval 后切分日志
        
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


class ImProgressBar(object):
    def __init__(self, total_iter, bar_len=50):
        self.total_iter = total_iter
        self.bar_len = bar_len
        self.coef = self.bar_len / 100
        self.foo = ['-', '\\', '|', '/']
        self.out_str = ''

    def update(self, i, msg=''):
        sys.stdout.write('\r')
        progress = int((i + 1) / self.total_iter * 100)
        self.out_str = "[%4s/%4s] %3s%% |%s%s| %s %s" % (
            (i + 1),
            self.total_iter,
            progress,
            int(progress * self.coef) * '>',
            (self.bar_len - int(progress * self.coef)) * ' ',
            self.foo[(i + 1) % len(self.foo)],
            msg
        )
        sys.stdout.write(self.out_str)
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write('\n')
        return self.out_str


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)             # CPU: Sets the seed for generating random numbers.
    torch.cuda.manual_seed_all(seed)    # GPU: Sets the seed for generating random numbers on all GPUs.
    np.random.seed(seed)                # numpy
    random.seed(seed)                   # random and transforms
    torch.backends.cudnn.deterministic = True   # cudnn


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def denormalize(tensor_img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    """
    Denormalize `tensor_img` with shape `(c, h, w)` with mean and std
    Return numpy `img` with shape `(h, w, c)`
    """
    img = tensor_img.numpy().transpose((1, 2, 0))
    img = img * std + mean
    return img


def tensor2np(tensor_img):
    return tensor_img.numpy().transpose((1, 2, 0))


def save_model_param(model, path):
    torch.save(model.state_dict(), path)


def load_model_param(model, path, device):
    saved_params = torch.load(path, map_location=device)
    model.load_state_dict(saved_params)
    model = model.to(device)
    return model


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    return torch.load(path)


def show_hists(r, c, figsize, datas, titles):
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=figsize)
    if r > 1:
        axes = list(itertools.chain.from_iterable(axes))
    for i in range(len(datas)):
        axes[i].set_title('{} {}'.format(np.array(datas[i]).shape, titles[i]))
        axes[i].hist(x=datas[i].view(-1).cpu().numpy())
    plt.show()


def show_imgs(r, c, figsize, imgs, titles, cmaps=None, colorbar=False, statistic=False):
    if cmaps is None:
        cmaps = [None] * len(imgs)
    fig, axes = plt.subplots(nrows=r, ncols=c, figsize=figsize)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0)
    if r > 1:
        axes = list(itertools.chain.from_iterable(axes))
    for i in range(len(axes)):
        axes[i].axis('off')
    for i in range(len(imgs)):
        axes[i].set_title('{} {}{}'.format(
            np.array(imgs[i]).shape, titles[i],
            '\nmin: {:.4f} max: {:.4f}\nmean: {:.4f} std: {:.4f}'.format(
                imgs[i].min(), imgs[i].max(), imgs[i].mean(), imgs[i].std()
            ) if statistic else ''
        ))
        ax = axes[i].imshow(imgs[i], interpolation='none', cmap=cmaps[i])
        if colorbar:
            fig.colorbar(ax, ax=axes[i])
    plt.show()


def show_tensor_img(img, title=''):
    _, h, w = img.shape
    plt.figure(figsize=(w / 150, h / 150), dpi=100)
    plt.title(title)
    plt.imshow(np.transpose(img.numpy(), (1,2,0)), interpolation='none')
    plt.show()


def show_img_grid(img_paths):
    trans = transforms.Compose([
        transforms.Resize(size=(224, 244)),
        transforms.ToTensor()
    ])
    imglist = [trans(Image.open(p).convert('RGB')) for p in img_paths]
    show_tensor_img(make_grid(imglist, padding=2))


def show_curve(ys, ylegends, title='Loss', xlable='Epoch', ylabel='Loss', ylog=False, legend_title=''):
    x = np.array(range(len(ys[0])))
    ys = [np.array(y) for y in ys]
    for i, y in enumerate(ys):
        plt.plot(x, y, label=ylegends[i])
    plt.axis()
    plt.title('{}'.format(title))
    plt.xlabel(xlable)
    plt.ylabel('{}'.format(ylabel))
    if ylog:
        plt.yscale("log")
    plt.legend(loc='best', title=legend_title)
    plt.grid()
    plt.show()
    # plt.savefig("{}.svg".format(ylabel))
    plt.close()


def get_data_from_log(path='./exp_pre/0918-202711/train.log', reg=r'Loss: (.*)]', reg_type=None):
    """
    e.g.
    r'Loss: (.*)]'
    r'Val MAE: (.*),'
    r'samples, lr (.*)'
    """
    if reg_type == 'loss':
        reg = r'Loss: (.*)]'
    elif reg_type == 'mae':
        reg = r'Val MAE: (.*),'
    elif reg_type == 'lr':
        reg = r'samples, lr (.*)'
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            cur_data = re.findall(reg, line)
            if len(cur_data) > 0:
                data.append(float(cur_data[0]))
    return data


def summary(model, input_size, batch_size=-1, device="cuda"):
    format_string = 'Input Shape: {}\n'.format([batch_size] + list(input_size))
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    format_string += "----------------------------------------------------------------\n"
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    format_string += line_new + '\n'
    format_string += "================================================================\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        format_string += line_new + '\n'

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    format_string += "================================================================\n"
    format_string += "Total params: {0:,}\n".format(total_params)
    format_string += "Trainable params: {0:,}\n".format(trainable_params)
    format_string += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    format_string += "----------------------------------------------------------------\n"
    format_string += "Input size (MB): %0.2f\n" % total_input_size
    format_string += "Forward/backward pass size (MB): %0.2f\n" % total_output_size
    format_string += "Params size (MB): %0.2f\n" % total_params_size
    format_string += "Estimated Total Size (MB): %0.2f\n" % total_size
    format_string += "----------------------------------------------------------------\n"
    return format_string


def to_str_lr_scheduler(scheduler):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        format_string = scheduler.__class__.__name__ + ' (\n'
        for param in ['mode', 'factor', 'patience', 'threshold', 'threshold_mode', 'cooldown', 'min_lrs', 'eps', 'verbose']:
            format_string += '    {0}: {1}\n'.format(param, getattr(scheduler, param))
        format_string += ')'
        return format_string
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        raise NotImplementedError
    else:
        return None


def to_str_args(args):
    format_string = args.__class__.__name__ + ' (\n'
    for k, v in vars(args).items():
        if isinstance(v, str):
            v = '\'{}\''.format(v)
        format_string += '    {0}: {1}\n'.format(k, v)
    format_string += ')'
    return format_string


if __name__ == "__main__":
    pass
