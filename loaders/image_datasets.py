import os.path
import torch
import os

from config.network_config import NetworkConfig

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional
import global_config
import kornia
from pathlib import Path
import kornia.augmentation as K

class GenericImageDataset(data.Dataset):
    def __init__(self, img_length, rgb_list, exr_list, segmentation_list, transform_config):
        network_config = NetworkConfig.getInstance().get_network_config()

        self.img_length = img_length
        self.rgb_list = rgb_list
        self.exr_list = exr_list
        self.segmentation_list = segmentation_list
        self.transform_config = transform_config

        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        if (self.transform_config == 1):
            patch_size = network_config["patch_size"]
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = (256, 256)

    def __getitem__(self, idx):
        try:
            rgb_img = cv2.imread(self.rgb_list[idx])
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = self.initial_op(rgb_img)

            depth_img = cv2.imread(self.exr_list[idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
            depth_img = self.initial_op(depth_img)

            segmentation_img = cv2.imread(self.segmentation_list[idx])
            segmentation_img = cv2.cvtColor(segmentation_img, cv2.COLOR_BGR2RGB)
            segmentation_img = self.initial_op(segmentation_img)

            if (self.transform_config == 1):
                crop_indices = transforms.RandomCrop.get_params(rgb_img, output_size=self.patch_size)
                i, j, h, w = crop_indices

                rgb_img = transforms.functional.crop(rgb_img, i, j, h, w)
                depth_img = transforms.functional.crop(depth_img, i, j, h, w)
                segmentation_img = transforms.functional.crop(segmentation_img, i, j, h, w)

            rgb_img = self.norm_op(rgb_img)
            depth_img = self.norm_op(depth_img)
            segmentation_img = self.norm_op(segmentation_img)

        except Exception as e:
            print("Failed to load: ", self.rgb_list[idx])
            print("ERROR: ", e)
            rgb_img = None
            depth_img = None
            segmentation_img = None

        return rgb_img, depth_img, segmentation_img

    def __len__(self):
        return self.img_length