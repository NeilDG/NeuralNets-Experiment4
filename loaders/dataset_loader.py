import glob
import random
import torch

import global_config
from config.network_config import ConfigHolder
from loaders import image_datasets
from torch.utils import data

def load_train_dataset(rgb_path, exr_path, segmentation_path):
    network_config = ConfigHolder.getInstance().get_network_config()
    general_config = global_config.general_config
    server_config = global_config.server_config
    exr_list = glob.glob(exr_path)
    rgb_list = glob.glob(rgb_path)
    segmentation_list = glob.glob(segmentation_path)

    for i in range(0, network_config["dataset_repeats"]): #TEMP: formerly 0-1
        rgb_list += rgb_list
        exr_list += exr_list
        segmentation_list += segmentation_list

    temp_list = list(zip(rgb_list, exr_list, segmentation_list))
    random.shuffle(temp_list)

    rgb_list, exr_list, segmentation_list = zip(*temp_list)
    img_length = len(rgb_list)
    print("Length of images: %d %d %d"  % (img_length, len(exr_list), len(segmentation_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.GenericImageDataset(img_length, rgb_list, exr_list, segmentation_list, 1),
        batch_size=network_config["load_size"][server_config],
        num_workers=general_config["num_workers"],
        shuffle=False
    )

    return data_loader, len(rgb_list)

def load_test_dataset(rgb_path, exr_path, segmentation_path):
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
        image_datasets.GenericImageDataset(img_length, rgb_list, exr_list, segmentation_list, 2),
        batch_size=general_config["test_size"],
        num_workers=2,
        shuffle=False
    )

    return data_loader, len(rgb_list)

def load_train_img2img_dataset(a_path, b_path):
    network_config = ConfigHolder.getInstance().get_network_config()
    general_config = global_config.general_config
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)
    a_list_dup = glob.glob(a_path)
    b_list_dup = glob.glob(b_path)

    print("Img to load is? " ,global_config.img_to_load)
    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]
        a_list_dup = a_list_dup[0: global_config.img_to_load]
        b_list_dup = b_list_dup[0: global_config.img_to_load]

    for i in range(0, network_config["dataset_a_repeats"]): #TEMP: formerly 0-1
        a_list += a_list_dup

    for i in range(0, network_config["dataset_b_repeats"]): #TEMP: formerly 0-1
        b_list += b_list_dup

    random.shuffle(a_list)
    random.shuffle(b_list)

    img_length = len(a_list)
    print("Length of images: %d %d"  % (img_length, len(b_list)))

    num_workers = int(general_config["num_workers"] / 2)
    data_loader_a = torch.utils.data.DataLoader(
        image_datasets.SingleImageDataset(a_list, 1),
        batch_size=global_config.load_size,
        num_workers=num_workers
        # shuffle=False,
        # pin_memory=True,
        # pin_memory_device=global_config.general_config["cuda_device"]
    )

    data_loader_b = torch.utils.data.DataLoader(
        image_datasets.SingleImageDataset(b_list, 1),
        batch_size=global_config.load_size,
        num_workers=num_workers
        # shuffle=False,
        # pin_memory=True,
        # pin_memory_device=global_config.general_config["cuda_device"]
    )

    return data_loader_a, data_loader_b, img_length

def load_test_img2img_dataset(a_path, b_path):
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    random.shuffle(a_list)
    random.shuffle(b_list)

    img_length = len(a_list)
    print("Length of images: %d %d" % (img_length, len(b_list)))

    data_loader_a = torch.utils.data.DataLoader(
        image_datasets.SingleImageDataset(a_list, 2),
        batch_size=global_config.load_size,
        num_workers=1,
        shuffle=False,
    )

    data_loader_b = torch.utils.data.DataLoader(
        image_datasets.SingleImageDataset(b_list, 2),
        batch_size=global_config.load_size,
        num_workers=1,
        shuffle=False,
    )

    return data_loader_a, data_loader_b, img_length

def load_kitti_test_dataset(rgb_path, depth_path):
    general_config = global_config.general_config

    rgb_list = glob.glob(rgb_path)
    depth_list = glob.glob(depth_path)

    temp_list = list(zip(rgb_list, depth_list))
    random.shuffle(temp_list)

    rgb_list, depth_list = zip(*temp_list)
    img_length = len(rgb_list)
    print("Length of images: %d %d" % (img_length, len(depth_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.KittiDepthDataset(img_length, rgb_list, depth_list),
        batch_size=general_config["test_size"],
        num_workers=2,
        shuffle=False
    )

    return data_loader, len(rgb_list)

