import os
from glob import glob

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from utils import (Model_Logger, RandomRotation90, _setup_size,
                   get_image_filename_list, random_crop)

IMG_EXTENSIONS = [
    '*.png', '*.jpeg', '*.jpg', '*.tif', '*.PNG', '*.JPEG', '*.JPG', '*.TIF'
]

logger = Model_Logger('data')


# Implement Dataset Class
class Counting_dataset(data.Dataset):
    def __init__(self,
                 root: str,
                 input_type: str,
                 crop_size: tuple | None = None,
                 transform: bool = True,
                 resize: tuple = None,
                 training_factor: int = 100,
                 memory_saving: bool = False):

        if input_type not in ['h5py', 'image']:
            raise TypeError("{} is not yet supported.".format(input_type))
        if input_type == 'h5py' and memory_saving is True:
            logger.warn(
                "When memory saving mode is not available for H5 type dataset. Set as False."
            )
            memory_saving = False
        self.root = root
        self.memory_saving = memory_saving
        self.C_size = _setup_size(crop_size, 'Error random crop size.')
        self.resize = _setup_size(resize, 'Error resize size.')
        self.type = input_type
        self.mode = None
        self.factor = training_factor
        self.transform = transform
        self.load_data()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.memory_saving:
            img = np.array(cv2.imread(self.imgs[index]))
            dot = np.array(cv2.imread(self.dots[index]))
        else:
            img = np.copy(self.imgs[index])
            dot = np.copy(self.dots[index])
        if len(img.shape) == 3 and img.shape[2] == 3:  # Check if img is a color image with 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(dot.shape) == 3 and dot.shape[2] == 3:
            dot = cv2.cvtColor(dot, cv2.COLOR_BGR2GRAY)
            
        # img.shape = (256, 256)

        img = img.astype(np.uint8)
        dot = dot.astype(np.float32)
        img = torch.Tensor(img)
        dot = torch.Tensor(dot)

        if self.transform and self.mode == 'train':
            img_dot = torch.concat(
                [img.unsqueeze(-1), dot.unsqueeze(-1)],
                dim=-1).permute(2, 0, 1)

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                RandomRotation90(90),
            ])
            img_dot = transform(img_dot).permute(1, 2, 0)
            img = img_dot[:, :, 0]
            dot = img_dot[:, :, 1]

        # Random cropping
        if self.C_size:
            if self.mode == 'train' and min(img.shape) >= self.C_size[0]:
                i, j, height, width = random_crop(img.shape, self.C_size)
                img = transforms.functional.crop(img, i, j, height, width)
                dot = transforms.functional.crop(dot, i, j, height, width)
            else:
                img = transforms.functional.resize(img.unsqueeze(0),
                                                   self.C_size,
                                                   antialias=True).squeeze(0)
                dot = transforms.functional.resize(dot.unsqueeze(0),
                                                   self.C_size,
                                                   antialias=True).squeeze(0)

        return img.unsqueeze(0), dot.unsqueeze(0)

    def load_data(self):
        if self.type == 'h5py':
            try:
                filename = glob(pathname='*.hdf5', root_dir=self.root)[0]
            except IndexError:
                raise FileNotFoundError(
                    "hdf5 file not found in {}. Check dataset type and file.".
                    format(self.root))
            h5_file = h5py.File(os.path.join(self.root, filename))
            try:
                self.imgs = np.asarray(h5_file['imgs'])
                self.dots = np.asarray(h5_file['counts'])
            except KeyError:
                raise KeyError(
                    "Not a proper hdf5 structure. It should include \'imgs\' key and \'counts\' key!"
                )
        else:  # type == 'image'
            file_list = get_image_filename_list(self.root)
            if file_list is None:
                raise FileNotFoundError(
                    "There are not any supported image files in {}".format(
                        self.root))
            imgs = []
            dots = []
            for dot_filename, raw_filename in file_list:
                dot_filename = os.path.join(self.root, 'dot', dot_filename)
                raw_filename = os.path.join(self.root, 'raw', raw_filename)
                if os.path.exists(raw_filename) is not True:
                    logger.warning(
                        "Could not find the raw file {}. Skipping this file"
                        .format(raw_filename))
                    continue
                if os.path.exists(dot_filename) is not True:
                    logger.warning(
                        "Could not find the annotations file {}. Skipping this file"
                        .format(dot_filename))
                    continue
                if self.memory_saving:
                    imgs.append(raw_filename)
                    dots.append(dot_filename)
                else:
                    imgs.append(np.asarray(cv2.imread(raw_filename)))
                    dots.append(np.asarray(cv2.imread(dot_filename)))
            self.imgs = imgs
            self.dots = dots

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

class Read_pseudolabel(data.Dataset):
    def __init__(self,
                 root: str,
                 input_type: str):
        self.root = root
        self.type = input_type
        self.load_data()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.copy(self.imgs[index])
        dot = np.copy(self.dots[index])
        
        img = img.astype(np.uint8)
        dot = dot.astype(np.float32)
        img = torch.Tensor(img)
        dot = torch.Tensor(dot)

        return img.unsqueeze(0), dot.unsqueeze(0)

    def load_data(self):
        if self.type == 'h5py':
            try:
                filename = glob(pathname='*.hdf5', root_dir=self.root)[0]
            except IndexError:
                raise FileNotFoundError(
                    "hdf5 file not found in {}. Check dataset type and file.".
                    format(self.root))
            h5_file = h5py.File(os.path.join(self.root, filename))
            try:
                self.imgs = np.asarray(h5_file['imgs'])
                self.dots = np.asarray(h5_file['counts'])
            except KeyError:
                raise KeyError(
                    "Not a proper hdf5 structure. It should include \'imgs\' key and \'counts\' key!"
                )
        else:  # type == 'image'
            file_list = get_image_filename_list(self.root)
            if file_list is None:
                raise FileNotFoundError(
                    "There are not any supported image files in {}".format(
                        self.root))
            imgs = []
            dots = []
            for dot_filename, raw_filename in file_list:
                dot_filename = os.path.join(self.root, 'dot', dot_filename)
                raw_filename = os.path.join(self.root, 'raw', raw_filename)
                if os.path.exists(raw_filename) is not True:
                    logger.warning(
                        "Could not find the raw file {}. Skipping this file"
                        .format(raw_filename))
                    continue
                if os.path.exists(dot_filename) is not True:
                    logger.warning(
                        "Could not find the annotations file {}. Skipping this file"
                        .format(dot_filename))
                    continue
                if self.memory_saving:
                    imgs.append(raw_filename)
                    dots.append(dot_filename)
                else:
                    imgs.append(np.asarray(cv2.imread(raw_filename)))
                    dots.append(np.asarray(cv2.imread(dot_filename)))
            self.imgs = imgs
            self.dots = dots

    def train(self):
        self.mode = 'train'

class MNIST_dataset(data.Dataset):
    def __init__(self,
                 dataset_dictionary: str,
                 filelist: list[str],
                 transform: tuple | None = None):
        self.root = dataset_dictionary
        self.transform = transform
        self.filenames = filelist
        self.load_data()

    def __len__(self):
        return self.labels.__len__()

    def __getitem__(self, index):
        img = np.copy(self.imgs[index])
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)

        return img.float(), torch.from_numpy(label).long()

    def load_data(self):
        self.imgs = []
        self.labels = []
        with open(self.filenames, 'r') as f:
            filelist = f.readlines()
        img_dir = os.path.basename(self.filenames)
        for filename in filelist:
            img_filename, label = filename.split(' ')
            img = np.asarray(
                cv2.imread(os.path.join(self.root, img_dir, img_filename)))
            if self.resize is not None:
                img = np.resize(img, (self.resize, self.resize))
            self.imgs.append(img)
            self.labels.append(np.asarray([int(label)]))
