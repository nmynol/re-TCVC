import os
import cv2
import torch
import numpy as np
from skimage import color, feature, util
from PIL import Image


def save_state(state, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    print("=> saving checkpoint of epoch " + str(epoch))
    torch.save(state, os.path.join(path, 'params_' + str(epoch) + '.pth'))
    print("saving completed.")


def load_state(path, netG, netD, optimizerG, optimizerD):
    assert os.path.isfile(path)
    print("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path)
    netG.load_state_dict(checkpoint['state_dictG'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    netD.load_state_dict(checkpoint['state_dictD'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    epoch = checkpoint['epoch'] + 1
    count = checkpoint['count']
    print("loading completed.")
    return epoch, count


def log_train_data(loginfo, opt):
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/" + opt.logfile
    with open(log_file, 'a') as f:
        f.write('%s\n' % ' '.join([str(item[1]) for item in loginfo]))


def sample(epoch, iteration, count, opt, sample_iterator, netG, device, writer):
    with torch.no_grad():

        input, target, prev_frame = next(sample_iterator)
        input, target, prev_frame = input.to(device), target.to(device), prev_frame.to(device)

        prediction = netG(torch.cat((input, prev_frame), 1))
        prediction = postprocess(prediction)
        input = postprocess(input)
        target = postprocess(target)

    img = stitch_images(input, target, prediction)
    # samples_dir = opt.sample_path
    #
    # if not os.path.exists(samples_dir):
    #     os.makedirs(samples_dir)

    sample = "sample" + "_" + str(epoch) + "_" + str(iteration).zfill(5) + ".png"
    print('\nsaving sample ' + sample + ' - learning rate: ' + str(opt.lr))
    # img.save(os.path.join(samples_dir, sample))
    writer.add_image("sample_image", np.array(img) / 255, count, dataformats='HWC')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    img = img.resize((256, 256))
    return img


def to_lineart(target):
    input = color.rgb2gray(np.array(target))
    input = feature.canny(input, sigma=1)
    input = (input == False)
    return input


def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img
