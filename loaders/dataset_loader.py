import glob
import random
import torch
from loaders import image_datasets
from torch.utils import data

def load_test_dataset(rgb_path, exr_path, segmentation_path, opts):
    exr_list = glob.glob(exr_path)
    rgb_list = glob.glob(rgb_path)
    segmentation_list = glob.glob(segmentation_path)

    temp_list = list(zip(rgb_list, exr_list, segmentation_list))
    random.shuffle(temp_list)

    rgb_list, exr_list, segmentation_list = zip(*temp_list)
    img_length = len(rgb_list)
    print("Length of images: %d %d %d"  % (img_length, len(exr_list), len(segmentation_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.GenericImageDataset(img_length, rgb_list, exr_list, segmentation_list, opts["train_config"]),
        batch_size=opts["load_size"],
        num_workers=opts["num_workers"],
        shuffle=False,
        pin_memory=True,
        pin_memory_device=opts["cuda_device"]
    )

    return data_loader

