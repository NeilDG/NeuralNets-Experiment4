# -*- coding: utf-8 -*-
import os

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

# dictionary keys
G_LOSS_KEY = "g_loss"
IDENTITY_LOSS_KEY = "id"
CYCLE_LOSS_KEY = "cyc"
G_ADV_LOSS_KEY = "g_adv"
LIKENESS_LOSS_KEY = "likeness"
LPIP_LOSS_KEY = "lpip"
SSIM_LOSS_KEY = "ssim"

D_OVERALL_LOSS_KEY = "d_loss"
D_A_REAL_LOSS_KEY = "d_real_a"
D_A_FAKE_LOSS_KEY = "d_fake_a"
D_B_REAL_LOSS_KEY = "d_real_b"
D_B_FAKE_LOSS_KEY = "d_fake_b"

LAST_METRIC_KEY = "last_metric"

plot_enabled = 1
disable_progress_bar = False

#Running on local = 0, Running on COARE = 1, Running on CCS server = 2
server_config = 0
general_config = {
    "num_workers" : 12,
    "cuda_device" : "cuda:0",
    "network_version" : "VXX.XX",
    "iteration" : 1,
    "current_epoch" : 0
}

path = ""
exr_path = ""
rgb_path = ""
segmentation_path = ""