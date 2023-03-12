from config import network_config
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
from testers import depth_metrics

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
        self.rmse_results = []
        self.rmse_log_results = []

    #measures the performance of a given batch and stores it
    def measure_and_store(self, input_map):
        rgb2target = self.dt.test(input_map)
        target_depth = input_map["depth"]

        rgb2target = tensor_utils.normalize_to_01(rgb2target)
        target_depth = tensor_utils.normalize_to_01(target_depth)
        depth_mask = torch.isfinite(target_depth) & (target_depth > 0.0)

        rgb2target = torch.masked_select(rgb2target, depth_mask)
        target_depth = torch.masked_select(target_depth, depth_mask)

        l1_result = self.l1_loss(rgb2target, target_depth).cpu()
        self.l1_results.append(l1_result)

        mse_result = self.mse_loss(rgb2target, target_depth).cpu()
        self.mse_results.append(mse_result)

        # num_zeros_target = np.shape(torch.flatten(target_depth))[0] - torch.count_nonzero(target_depth).item()
        # num_zeros_pred = np.shape(torch.flatten(rgb2target))[0] - torch.count_nonzero(rgb2target).item()
        # print("Has zeros? ", num_zeros_pred, num_zeros_target)

        rmse_result = depth_metrics.torch_rmse(rgb2target, target_depth).cpu()
        self.rmse_results.append(rmse_result)

        rmse_log_result = depth_metrics.torch_rmse_log(target_depth,rgb2target).cpu()
        print("Rmse log result: ", rmse_log_result)
        self.rmse_log_results.append(rmse_log_result)

    def report_and_visualize(self, input_map, dataset_title):
        version_name = network_config.NetworkConfig.getInstance().get_version_name()

        self.dt.visdom_visualize(input_map, "Test - " + version_name + " " + dataset_title)

        l1_mean = np.round(np.mean(self.l1_results), 4)
        self.l1_results.clear()

        mse_mean = np.round(np.mean(self.mse_results), 4)
        self.mse_results.clear()

        rmse_mean = np.round(np.mean(self.rmse_results), 4)
        self.rmse_results.clear()

        rmse_log_mean = np.round(np.mean(self.rmse_log_results), 4)
        self.rmse_log_results.clear()

        self.visdom_reporter.plot_text(dataset_title + " Results - " + version_name + "<br>"
                                       + "Abs Rel: " + str(l1_mean) + "<br>"
                                        "Sqr Rel: " + str(mse_mean) + "<br>"
                                       "RMSE: " + str(rmse_mean) + "<br>"
                                       "RMSE log: " +str(rmse_log_mean) +"<br>")