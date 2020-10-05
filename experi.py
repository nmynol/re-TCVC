import os
import torch
import numpy as np
import argparse
import random
import yaml
from easydict import EasyDict
import gensim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter

from models.standard import *
from util.utils import *
from util.loss import *
from util.data import *


def set_parser():
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--train_set', default='/data2/wn/Video_dataset/train/frame', help='facades')
    parser.add_argument('--test_set', default='/data2/wn/Video_dataset/test/frame', help='facades')
    parser.add_argument('--logfile', default='./log', help='trainlogs.dat')
    parser.add_argument('--log_freq', type=int, default=25, help='log frequency (/iteration)')
    parser.add_argument('--sample_freq', type=int, default=250, help='sample frequency (/iteration)')
    parser.add_argument('--save_freq', type=int, default=25, help='save frequency (/epoch)')
    parser.add_argument('--sample_path', default="./samples", help='sample path')
    parser.add_argument('--board_path', default="./board1", help='tensorboard path')
    parser.add_argument('--checkpoint_path', default="", help='load pre-trained model?')
    parser.add_argument('--save_path', default="./param", help='the path to save the model to')
    parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
    parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='不知道是啥channel')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.002')
    parser.add_argument('--beta1', type=float, default=0, help='beta1 for adam. default=0.5')
    parser.add_argument('--workers', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=1275, help='random seed to use. Default=123')
    parser.add_argument('--L1lamb', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--Stylelamb', type=int, default=1000, help='weight on Style term in objective')
    parser.add_argument('--Contentlamb', type=int, default=1, help='weight on Content term in objective')
    parser.add_argument('--Adversariallamb', type=int, default=0.1, help='weight on Adv term in objective')
    return parser.parse_args()


if __name__ == '__main__':
    opt = set_parser()

    epoch = 1
    count = 0

    train_set = MyDataset("../dataset/re-TCVC/data/frame")

    data_loader = DataLoader(train_set, batch_size=1)
    sample_iterator = create_iterator(1, train_set)

    writer = SummaryWriter(opt.board_path)

    for epoch in range(epoch, opt.epoch + 1):
        for iteration, [input, real, prev] in enumerate(data_loader):
            # input, real, prev = next(sample_iterator)
            input = np.uint8(np.transpose(np.array(input.squeeze(0)), (1, 2, 0)) * 255)
            real = np.uint8(np.transpose(np.array(real.squeeze(0)), (1, 2, 0)) * 255)
            prev = np.uint8(np.transpose(np.array(prev.squeeze(0)), (1, 2, 0)) * 255)

            real = Image.fromarray(real)
            prev = Image.fromarray(prev)
            # real.show()
            # prev.show()
            assert False