import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from trainers import depth_trainer
from tqdm import tqdm
from tqdm.auto import trange
from time import sleep
import yaml
from yaml.loader import SafeLoader

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="vXX.XX")
parser.add_option('--iteration', type=int, default=1)
parser.add_option('--plot_enabled', type=int, default=1)
parser.add_option('--save_every_iter', type=int, default=400)

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled
    global_config.general_config["network_version"] = opts.network_version
    global_config.general_config["iteration"] = opts.iteration
    network_config = ConfigHolder.getInstance().get_network_config()

    if(global_config.server_config == 0): #COARE
        global_config.general_config["num_workers"] = 6
        global_config.disable_progress_bar = True
        global_config.path = "/scratch1/scratch2/neil.delgallego/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using COARE configuration. Workers: ", global_config.general_config["num_workers"])

    elif(global_config.server_config == 1): #CCS Cloud
        global_config.general_config["num_workers"] = 12
        global_config.path = "/home/jupyter-neil.delgallego/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using CCS configuration. Workers: ", global_config.general_config["num_workers"])

    elif(global_config.server_config == 2): #RTX 2080Ti
        global_config.general_config["num_workers"] = 6

        print("Using RTX 2080Ti configuration. Workers: ", global_config.general_config["num_workers"])

    elif(global_config.server_config == 3):
        global_config.general_config["num_workers"] = 12
        global_config.path = "X:/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using RTX 3090 configuration. Workers: ", global_config.general_config["num_workers"])

    global_config.path = global_config.path.format(dataset_version=network_config["dataset_version"])
    global_config.exr_path = global_config.path + "*.exr"
    global_config.rgb_path = global_config.path + "*.camera.png"
    global_config.segmentation_path = global_config.path + "*.semantic segmentation.png"


def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    hyperparam_config = ConfigHolder.getInstance().get_hyper_params()
    network_iteration = global_config.general_config["iteration"]
    hyperparams_table = hyperparam_config["hyperparams"][network_iteration]
    print("Network iteration:", str(network_iteration), ". Hyper parameters: ", hyperparams_table, " Learning rates: ", network_config["g_lr"], network_config["d_lr"])

    rgb_path = global_config.rgb_path
    exr_path = global_config.exr_path
    segmentation_path = global_config.segmentation_path

    print("Dataset path: ", rgb_path)

    plot_utils.VisdomReporter.initialize()
    visdom_reporter = plot_utils.VisdomReporter.getInstance()

    train_loader, train_count = dataset_loader.load_train_dataset(rgb_path, exr_path, segmentation_path)
    test_loader, test_count = dataset_loader.load_test_dataset(rgb_path, exr_path, segmentation_path)
    dt = depth_trainer.DepthTrainer(device)

    iteration = 0
    start_epoch = global_config.general_config["current_epoch"]
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: depth", " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    # compute total progress
    load_size = network_config["load_size"][global_config.server_config]
    needed_progress = int((network_config["max_epochs"]) * (train_count / load_size))
    current_progress = int(start_epoch * (train_count / load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    for epoch in range(start_epoch, network_config["max_epochs"]):
        for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
            rgb_batch, depth_batch, _ = train_data
            rgb_batch = rgb_batch.to(device)
            depth_batch = depth_batch.to(device)

            rgb_unseen, depth_unseen, _ = test_data #TODO: Change to cityscapes/KITTI
            rgb_unseen = rgb_unseen.to(device)
            depth_unseen = depth_unseen.to(device)

            input_map = {"rgb" : rgb_batch, "depth" : depth_batch, "rgb_unseen" : rgb_unseen, "depth_unseen" : depth_unseen}
            dt.train(epoch, iteration, input_map, input_map)

            iteration = iteration + 1
            pbar.update(1)

            if(dt.is_stop_condition_met()):
                break

            if(iteration % opts.save_every_iter == 0):
                dt.save_states(epoch, iteration, True)

                if(global_config.plot_enabled == 1):
                    dt.visdom_plot(iteration)
                    dt.visdom_visualize(input_map, "Train")

                    rgb_batch, depth_batch, _ = test_data
                    rgb_batch = rgb_batch.to(device)
                    depth_batch = depth_batch.to(device)
                    input_map = {"rgb": rgb_batch, "depth": depth_batch}

                    dt.visdom_visualize(input_map, "Test")

        if(dt.is_stop_condition_met()):
            break

    pbar.close()

if __name__ == "__main__":
    main(sys.argv)