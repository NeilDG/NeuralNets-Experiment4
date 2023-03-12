from config.network_config import NetworkConfig
from losses import depth_losses
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
from trainers import early_stopper

class DepthTrainer(abstract_iid_trainer.AbstractIIDTrainer):
    def __init__(self, gpu_device):
        super().__init__(gpu_device)
        self.initialize_train_config()

    def initialize_train_config(self):
        network_config = NetworkConfig.getInstance().get_network_config()
        general_config = global_config.general_config
        self.iteration = general_config["iteration"]
        self.hyperparams_table = network_config["hyperparams"][self.iteration]
        self.use_bce = self.hyperparams_table["is_bce"]
        self.adv_weight = self.hyperparams_table["adv_weight"]

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.gradient_loss = depth_losses.GradientLoss()
        self.ssi_loss = depth_losses.ScaleAndShiftInvariantLoss()

        self.D_SM_pool = image_pool.ImagePool(50)

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision
        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.load_size = network_config["load_size"]
        self.batch_size = network_config["batch_size"]

        self.stopper_method = early_stopper.EarlyStopper(network_config["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, 1000)
        self.stop_result = False

        self.initialize_dict()
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_depth, self.D_depth = network_creator.initialize_depth_network()

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_depth.parameters()), lr=self.g_lr, weight_decay=network_config["weight_decay"])
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_depth.parameters()), lr=self.d_lr, weight_decay=network_config["weight_decay"])
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1000000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = NetworkConfig.getInstance().get_version_name()
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            return self.mse_loss(pred, target)
        else:
            return self.bce_loss(pred, target)

    def l1_depth_log_loss(self, pred, target):
        loss = torch.mean(torch.abs(torch.log(pred) - torch.log(target)))
        return loss

    def lpip_loss(self, pred, target):
        result = torch.squeeze(self.lpips_loss(pred, target))
        result = torch.mean(result)
        return result

    def ssim_loss(self, pred, target):
        pred_normalized = (pred * 0.5) + 0.5
        target_normalized = (target * 0.5) + 0.5

        return self.ssim_loss(pred_normalized, target_normalized)

    #From monodepth2 by godard
    def get_smooth_loss(self, pred, target):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_disp_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def initialize_dict(self):

        # dictionary keys
        self.G_LOSS_KEY = "g_loss"
        self.IDENTITY_LOSS_KEY = "id"
        self.CYCLE_LOSS_KEY = "cyc"
        self.G_ADV_LOSS_KEY = "g_adv"
        self.LIKENESS_LOSS_KEY = "likeness"
        self.LPIP_LOSS_KEY = "lpip"
        self.SSIM_LOSS_KEY = "ssim"
        self.DISP_SMOOTH_LOSS_KEY = "disp_loss"
        self.GRADIENT_LOSS_KEY = "grad_loss"
        self.SSI_LOSS_KEY = "ssi_loss"

        self.D_OVERALL_LOSS_KEY = "d_loss"
        self.D_A_REAL_LOSS_KEY = "d_real_a"
        self.D_A_FAKE_LOSS_KEY = "d_fake_a"
        self.D_B_REAL_LOSS_KEY = "d_real_b"
        self.D_B_FAKE_LOSS_KEY = "d_fake_b"


        # what to store in visdom?
        self.losses_dict_s = {}
        self.losses_dict_s[self.G_LOSS_KEY] = []
        self.losses_dict_s[self.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_s[self.LIKENESS_LOSS_KEY] = []
        self.losses_dict_s[self.LPIP_LOSS_KEY] = []
        self.losses_dict_s[self.G_ADV_LOSS_KEY] = []
        self.losses_dict_s[self.DISP_SMOOTH_LOSS_KEY] = []
        self.losses_dict_s[self.GRADIENT_LOSS_KEY] = []
        self.losses_dict_s[self.SSI_LOSS_KEY] = []
        self.losses_dict_s[self.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_s[self.D_A_REAL_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[self.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict_s[self.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[self.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[self.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[self.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[self.DISP_SMOOTH_LOSS_KEY] = "Disp smooth loss per iteration"
        self.caption_dict_s[self.GRADIENT_LOSS_KEY] = "Gradient loss per iteration"
        self.caption_dict_s[self.SSI_LOSS_KEY] = "Scale-invariant loss per iteration"
        self.caption_dict_s[self.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[self.D_A_REAL_LOSS_KEY] = "D real loss per iteration"

        # what to store in visdom?
        self.losses_dict_t = {}

        self.TRAIN_LOSS_KEY = "TRAIN_LOSS_KEY"
        self.losses_dict_t[self.TRAIN_LOSS_KEY] = []
        self.TEST_LOSS_KEY = "TEST_LOSS_KEY"
        self.losses_dict_t[self.TEST_LOSS_KEY] = []

        self.caption_dict_t = {}
        self.caption_dict_t[self.TRAIN_LOSS_KEY] = "Train L1 loss per iteration"
        self.caption_dict_t[self.TEST_LOSS_KEY] = "Test L1 loss per iteration"

    def train(self, epoch, iteration, input_map, target_map):
        input_rgb = input_map["rgb"]
        target_tensor = target_map["depth"]

        accum_batch_size = self.load_size * iteration

        with amp.autocast():
            #discriminator
            self.optimizerD.zero_grad()
            self.D_depth.train()

            output = self.G_depth(input_rgb)
            prediction = self.D_depth(target_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_SM_real_loss = self.adversarial_loss(self.D_depth(target_tensor), real_tensor) * self.adv_weight
            D_SM_fake_loss = self.adversarial_loss(self.D_SM_pool.query(self.D_depth(output.detach())), fake_tensor) * self.adv_weight

            errD = D_SM_real_loss + D_SM_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.schedulerD.step(errD)
                self.fp16_scaler.step(self.optimizerD)

            # shadow map generator
            self.optimizerG.zero_grad()
            self.G_depth.train()
            rgb2target = self.G_depth(input_rgb)
            SM_likeness_loss = self.l1_loss(rgb2target, target_tensor) * self.hyperparams_table["l1_weight"]
            SM_lpip_loss = self.lpip_loss(rgb2target, target_tensor) * self.hyperparams_table["lpip_weight"]
            SM_smooth_loss = self.get_smooth_loss(rgb2target, target_tensor) * self.hyperparams_table["disp_weight"]
            mask_tensor = (target_tensor > 0.01) * 1.0
            SM_grad_loss = self.gradient_loss(rgb2target, target_tensor, mask_tensor) * self.hyperparams_table["grad_weight"]
            # SM_ssi_loss = self.ssi_loss(rgb2target, target_tensor, mask_tensor) * self.hyperparams_table["ssi_weight"]

            prediction = self.D_depth(rgb2target)
            real_tensor = torch.ones_like(prediction)
            SM_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = SM_likeness_loss + SM_lpip_loss + SM_smooth_loss + SM_grad_loss + SM_adv_loss

            self.fp16_scaler.scale(errG).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.schedulerG.step(errG)
                self.fp16_scaler.step(self.optimizerG)
                self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            if (iteration > 50):
                self.losses_dict_s[self.G_LOSS_KEY].append(errG.item())
                self.losses_dict_s[self.D_OVERALL_LOSS_KEY].append(errD.item())
                self.losses_dict_s[self.LIKENESS_LOSS_KEY].append(SM_likeness_loss.item())
                self.losses_dict_s[self.LPIP_LOSS_KEY].append(SM_lpip_loss.item())
                self.losses_dict_s[self.G_ADV_LOSS_KEY].append(SM_adv_loss.item())
                self.losses_dict_s[self.DISP_SMOOTH_LOSS_KEY].append(SM_smooth_loss.item())
                self.losses_dict_s[self.GRADIENT_LOSS_KEY].append(SM_grad_loss.item())
                # self.losses_dict_s[self.SSI_LOSS_KEY].append(SM_ssi_loss.item())
                self.losses_dict_s[self.D_A_FAKE_LOSS_KEY].append(D_SM_fake_loss.item())
                self.losses_dict_s[self.D_A_REAL_LOSS_KEY].append(D_SM_real_loss.item())

                # perform validation test and early stopping
                rgb2target_unseen = self.test_unseen(input_map)
                target_unseen = input_map["depth_unseen"]
                self.stopper_method.register_metric(rgb2target_unseen, target_unseen, epoch)
                self.stop_result = self.stopper_method.test(epoch)

                if (self.stopper_method.has_reset()):
                    self.save_states(epoch, iteration, False)

                # plot train-test loss
                rgb2target_test = self.test(input_map)
                rgb2target_unseen = tensor_utils.normalize_to_01(rgb2target_unseen)
                target_unseen = tensor_utils.normalize_to_01(target_unseen)
                rgb2target_test = tensor_utils.normalize_to_01(rgb2target_test)
                target_tensor = tensor_utils.normalize_to_01(target_tensor)
                self.losses_dict_t[self.TRAIN_LOSS_KEY].append(self.l1_loss(rgb2target_test, target_tensor).item())
                self.losses_dict_t[self.TEST_LOSS_KEY].append(self.l1_loss(rgb2target_unseen, target_unseen).item())

    def test(self, input_map):
        with torch.no_grad():
            self.G_depth.eval()

            input_rgb = input_map["rgb"]
            rgb2target = self.G_depth(input_rgb)

        return rgb2target

    def test_unseen(self, input_map):
        with torch.no_grad():
            self.G_depth.eval()

            input_rgb = input_map["rgb_unseen"]
            rgb2target = self.G_depth(input_rgb)

        return rgb2target

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_CHECKPATH)
        self.visdom_reporter.plot_train_test_loss("train_test_loss", iteration, self.losses_dict_t, self.caption_dict_t, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        input_rgb = input_map["rgb"]
        rgb2target = self.test(input_map)

        self.visdom_reporter.plot_image(input_rgb, str(label) + " RGB Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb2target, str(label) + " Depth-Like images - " + self.NETWORK_VERSION + str(self.iteration))
        if("depth" in input_map):
            target_tensor = input_map["depth"]
            self.visdom_reporter.plot_image(target_tensor, str(label) + " Depth images - " + self.NETWORK_VERSION + str(self.iteration))

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, global_config.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_depth.state_dict()
        netDNS_state_dict = self.D_depth.state_dict()

        save_dict[global_config.GENERATOR_KEY + "Z"] = netGNS_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + "Z"] = netDNS_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new depth network: ", self.NETWORK_CHECKPATH)

        if(checkpoint != None):
            global_config.general_config["current_epoch"] = checkpoint["epoch"]
            self.stopper_method.update_last_metric(checkpoint[global_config.LAST_METRIC_KEY])
            self.G_depth.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "Z"])
            self.D_depth.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY + "Z"])

            print("Loaded depth network: ", self.NETWORK_CHECKPATH, "Epoch: ", global_config.general_config["current_epoch"])
