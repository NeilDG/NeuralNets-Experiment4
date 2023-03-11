from config.network_config import NetworkConfig
from trainers import abstract_iid_trainer
import global_config
import torch
import torch.cuda.amp as amp
import itertools
from model.modules import image_pool
from utils import plot_utils, tensor_utils
import lpips
import torch.nn as nn
import numpy as np
from trainers import depth_trainer

class DepthTester():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.dt = depth_trainer.DepthTrainer(self.gpu_device)
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.l1_results = []
        self.mse_results = []

    #measures the performance of a given batch and stores it
    def measure_and_store(self, input_map):
        rgb2target = self.dt.test(input_map)

        target_depth = input_map["depth"]
        rgb2target = tensor_utils.normalize_to_01(rgb2target)
        target_depth = tensor_utils.normalize_to_01(target_depth)

        l1_result = self.l1_loss(rgb2target, target_depth).cpu()
        self.l1_results.append(l1_result)

        mse_result = self.mse_loss(rgb2target, target_depth).cpu()
        self.mse_results.append(mse_result)

    def report_and_visualize(self, input_map):
        self.dt.visdom_visualize(input_map, "Test")

        l1_mean = np.round(np.mean(self.l1_results), 4)
        self.l1_results.clear()

        mse_mean = np.round(np.mean(self.mse_results), 4)
        self.mse_results.clear()

        self.visdom_reporter.plot_text("L1 mean: " + str(l1_mean) + "\n"
                                        "MSE mean: " + str(mse_mean))
