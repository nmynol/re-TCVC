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
from PIL import Image
from models.standard import *
from util.utils import *
from util.loss import *
from util.data import *


def set_parser():
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--val_set', default='../dataset/re-TCVC/make_video/origin_pic', help='facades')
    parser.add_argument('--save_to', default='../dataset/re-TCVC/make_video/output_pic', help='facades')
    parser.add_argument('--checkpoint_path', default="../dataset/re-TCVC/make_video/param/params.pth", help='load pre-trained model?')
    parser.add_argument('--val_batch_size', type=int, default=1, help='validation batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use. Default=123')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.002')
    parser.add_argument('--beta1', type=float, default=0, help='beta1 for adam. default=0.5')
    return parser.parse_args()


if __name__ == '__main__':
    opt = set_parser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    netG = InpaintGenerator()
    netD = Discriminator(in_channels=7, use_sigmoid=True)

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr * 0.1, betas=(opt.beta1, 0.999))

    if opt.checkpoint_path:
        epoch, count = load_state(opt.checkpoint_path, netG, netD, optimizerG, optimizerD)

    netD = netD.to(device).eval()
    netG = netG.to(device).eval()

    val_set = MyDataset(opt.val_set)

    val_data_loader = DataLoader(val_set, batch_size=opt.val_batch_size,
                                 pin_memory=True, num_workers=opt.workers)

    transform_list = [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    for i in range(1800):
        target = load_img(os.path.join(opt.val_set, "frame0" + str(3201 + i) + ".jpg"))
        input = to_lineart(target)
        prev = load_img(os.path.join(opt.save_to, str(i) + ".jpg"))

        input = transform(input).type(torch.FloatTensor)
        prev = transform(prev).type(torch.FloatTensor)

        input, prev = input.to(device).unsqueeze(0), prev.to(device).unsqueeze(0)

        netG.eval()
        prediction = netG(torch.cat((input, prev), 1))

        prediction = prediction * 255.0
        prediction = np.array(prediction.squeeze().int().cpu()).astype(np.uint8)
        prediction = Image.fromarray(np.transpose(prediction, (1, 2, 0)))
        prediction.save(os.path.join(opt.save_to, str(i + 1) + ".jpg"))
        print("save image to:", os.path.join(opt.save_to, str(i + 1) + ".jpg"))
