import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .utils import is_image_file, load_img, to_lineart
from skimage import color, feature, util


class MyDataset1(data.Dataset):
    def __init__(self, image_dir):
        super(MyDataset1, self).__init__()
        self.path = image_dir
        self.image_filenames = [x for x in os.listdir(self.path) if is_image_file(x)]
        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def get_prev(self, num):
        if not os.path.exists(os.path.join(self.path, "frame" + str(num) + ".jpg")):
            initial_prev_frame = Image.new("RGB", [256, 256])
            return initial_prev_frame
        else:
            rnd = np.random.uniform(0, 1)
            if rnd <= 0.5:
                prev = load_img(os.path.join(self.path, "frame" + str(num) + ".jpg"))
            else:
                prev = Image.new("RGB", [256, 256])
            return prev

    def __getitem__(self, index):
        target_path = os.path.join(self.path, self.image_filenames[index])
        frame_num = target_path.split("e")[-1]
        frame_num = int(frame_num.split(".")[0]) - 1
        # will be either black or colored
        frame_prev = self.get_prev(frame_num)
        target = load_img(target_path)
        input = to_lineart(target)

        frame_prev = self.transform(frame_prev)
        target = self.transform(target)
        input = self.transform(input)

        return input.type(torch.FloatTensor), target.type(torch.FloatTensor), frame_prev.type(torch.FloatTensor)

    def __len__(self):
        return len(self.image_filenames)


class MyDataset(data.Dataset):
    def __init__(self, image_dir):
        super(MyDataset, self).__init__()
        self.path = image_dir
        self.image_filenames = [x for x in os.listdir(self.path) if is_image_file(x)]
        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def get_prev(self, name):
        if name.split('_')[-1].split('.')[0] == '0000':
            initial_prev_frame = Image.new("RGB", [256, 256])
            return initial_prev_frame
        else:
            rnd = np.random.uniform(0, 1)
            if rnd <= 0.5:
                prev = load_img(os.path.join(self.path, name))
            else:
                prev = Image.new("RGB", [256, 256])
            return prev

    def __getitem__(self, index):
        target_path = os.path.join(self.path, self.image_filenames[index])
        apart = self.image_filenames[index].split("_")
        frame_num = apart[-1].split('.')[0]
        prev_name = apart[0] + "_" + apart[1] + "_" + str(int(frame_num) - 1).zfill(4) + "." + apart[-1].split('.')[-1]
        # will be either black or colored
        frame_prev = self.get_prev(prev_name)
        target = load_img(target_path)
        input = to_lineart(target)

        frame_prev = self.transform(frame_prev)
        target = self.transform(target)
        input = self.transform(input)

        return input.type(torch.FloatTensor), target.type(torch.FloatTensor), frame_prev.type(torch.FloatTensor)

    def __len__(self):
        return len(self.image_filenames)


def create_iterator(sample_size, sample_dataset):
    while True:
        sample_loader = DataLoader(
            dataset= sample_dataset,
            batch_size=sample_size,
            drop_last=True
        )

        for item in sample_loader:
            yield item