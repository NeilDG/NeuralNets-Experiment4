import glob
import random
import torch

import global_config
from config.network_config import NetworkConfig
from loaders import image_datasets
from torch.utils import data

def load_custom_dataset(rgb_path, exr_path, segmentation_path, train_config):
    network_config = NetworkConfig.getInstance().get_network_config()
    general_config = global_config.general_config
    exr_list = glob.glob(exr_path)
    rgb_list = glob.glob(rgb_path)
    segmentation_list = glob.glob(segmentation_path)

    temp_list = list(zip(rgb_list, exr_list, segmentation_list))
    random.shuffle(temp_list)

    rgb_list, exr_list, segmentation_list = zip(*temp_list)
    img_length = len(rgb_list)
    print("Length of images: %d %d %d"  % (img_length, len(exr_list), len(segmentation_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.GenericImageDataset(img_length, rgb_list, exr_list, segmentation_list, train_config),
        batch_size=network_config["load_size"],
        num_workers=general_config["num_workers"],
        shuffle=False,
        pin_memory=True,
        pin_memory_device=general_config["cuda_device"]
    )

    return data_loader, len(rgb_list)

