import math
import os
import torch.nn as nn
from progressbar import ETA, Bar, Percentage, ProgressBar

from .dataset import shuffle_data
from .disco_gan_model import DiscoGAN, get_generators, get_discriminators
from .error_bound_calc_functions import calc_mean_gt_error, calc_mean_error_bound
from .utils import *


class DiscoGANRisk(DiscoGAN):
    def __init__(self, options):
        super().__init__(options)

        self.result_path_1 = os.path.join(self.result_path, 'G1')
        self.result_path_2 = os.path.join(self.result_path, 'G2')
        os.makedirs(self.result_path_1, exist_ok=True)
        os.makedirs(self.result_path_2, exist_ok=True)
        
        lr = self.args.learning_rate

        if self.args.start_from_pretrained_g1:
            gen_A_path = os.path.join(self.args.pretrained_g1_path_A, 'model_gen_A_G1-' + str(self.args.which_epoch_load))
            self.generator_A = torch.load(gen_A_path)
            gen_B_path = os.path.join(self.args.pretrained_g1_path_B, 'model_gen_B_G1-' + str(self.args.which_epoch_load))
            self.generator_B = torch.load(gen_B_path)
            dis_A_path = os.path.join(self.args.pretrained_g1_path_A, 'model_dis_A_G1-' + str(self.args.which_epoch_load))
            self.discriminator_A = torch.load(dis_A_path)
            dis_B_path = os.path.join(self.args.pretrained_g1_path_B, 'model_dis_B_G1-' + str(self.args.which_epoch_load))
            self.discriminator_B = torch.load(dis_B_path)
            _, _, self.optim_gen = get_generators(self.cuda, self.args.num_layers, lr)
            _, _, self.optim_dis = get_discriminators(self.cuda, lr)
        else:
            self.generator_A, self.generator_B, self.optim_gen = get_generators(self.cuda, self.args.num_layers, lr)
            self.discriminator_A, self.discriminator_B, self.optim_dis = get_discriminators(self.cuda, lr)
        
        if self.args.start_from_pretrained_g2:
            gen_A_path = os.path.join(self.args.pretrained_g2_path_A, 'model_gen_A_G2-' + str(self.args.which_epoch_load))
            self.generator_A_G2 = torch.load(gen_A_path)
            gen_B_path = os.path.join(self.args.pretrained_g2_path_B, 'model_gen_B_G2-' + str(self.args.which_epoch_load))
            self.generator_B_G2 = torch.load(gen_B_path)
            dis_A_path = os.path.join(self.args.pretrained_g2_path_A, 'model_dis_A_G2-' + str(self.args.which_epoch_load))
            self.discriminator_A_G2 = torch.load(dis_A_path)
            dis_B_path = os.path.join(self.args.pretrained_g2_path_B, 'model_dis_B_G2-' + str(self.args.which_epoch_load))
            self.discriminator_B_G2 = torch.load(dis_B_path)
            _, _, self.optim_gen_G2 = get_generators(self.cuda, self.args.num_layers, lr)
            _, _, self.optim_dis_G2 = get_discriminators(self.cuda, lr)
        else:
            self.generator_A_G2, self.generator_B_G2, self.optim_gen_G2 = get_generators(self.cuda, self.args.num_layers, lr)
            self.discriminator_A_G2, self.discriminator_B_G2, self.optim_dis_G2 = get_discriminators(self.cuda, lr)
        
        self.correlation_criterion = nn.L1Loss()

    def _save_model(self):
        version = str(int(self.iters / self.args.model_save_interval))
        torch.save(self.generator_A, os.path.join(self.model_path, 'model_gen_A_G1-' + version))
        torch.save(self.generator_B, os.path.join(self.model_path, 'model_gen_B_G1-' + version))
        torch.save(self.discriminator_A, os.path.join(self.model_path, 'model_dis_A_G1-' + version))
        torch.save(self.discriminator_B, os.path.join(self.model_path, 'model_dis_B_G1-' + version))
        torch.save(self.generator_A_G2, os.path.join(self.model_path, 'model_gen_A_G2-' + version))
        torch.save(self.generator_B_G2, os.path.join(self.model_path, 'model_gen_B_G2-' + version))
        torch.save(self.discriminator_A_G2, os.path.join(self.model_path, 'model_dis_A_G2-' + version))
        torch.save(self.discriminator_B_G2, os.path.join(self.model_path, 'model_dis_B_G2-' + version))
        print('Models checkpoint saved at {0}, version {1}'.format(self.model_path, version))

    def _log_losses(self, A, B):
        super()._log_losses(A, B)
        A, B = read_images(A, B, self.args.image_size, self.cuda, aligned=True)
        loss = nn.L1Loss()
        gt_error_A_G2 = calc_mean_gt_error(B, A, self.generator_A_G2, loss)
        gt_error_B_G2 = calc_mean_gt_error(A, B, self.generator_B_G2, loss)
        self.board_writer.add_scalar('GT Error A G2', gt_error_A_G2, self.iters)
        self.board_writer.add_scalar('GT Error B G2', gt_error_B_G2, self.iters)
        error_bound_A = calc_mean_error_bound(B, self.generator_A, self.generator_A_G2, loss)
        error_bound_B = calc_mean_error_bound(A, self.generator_B, self.generator_B_G2, loss)
        self.board_writer.add_scalar('Bound A', error_bound_A, self.iters)
        self.board_writer.add_scalar('Bound B', error_bound_B, self.iters)

    def _calc_corr_loss(self, A, B):
        AB_1 = self.generator_B(A)
        BA_1 = self.generator_A(B)
        AB_2 = self.generator_B_G2(A)
        BA_2 = self.generator_A_G2(B)

        # Correlation loss
        # Distance between generator 1 and generator 2's output
        correlation_loss_AB_2 = - self.correlation_criterion(AB_2, to_no_grad_var(AB_1))
        correlation_loss_BA_2 = - self.correlation_criterion(BA_2, to_no_grad_var(BA_1))
        return correlation_loss_AB_2, correlation_loss_BA_2

    def train(self, data_A, data_B, data_A_val, data_B_val):
        n_batches = math.ceil((len(data_A)) / self.args.batch_size)
        print('%d batches per epoch' % n_batches)
        self.iters = 0

        for epoch in range(self.args.epoch_size):
            A_epoch, B_epoch = shuffle_data(data_A, data_B)

            widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=n_batches, widgets=widgets)
            pbar.start()

            for i in range(n_batches):
                pbar.update(i)

                # read batch data
                A_paths, B_paths = get_batch_data(A_epoch, B_epoch, i, self.args.batch_size)
                A, B = read_images(A_paths, B_paths, self.args.image_size, self.cuda)

                # calculate losses
                if self.iters < self.args.gan_curriculum:
                    rate = self.args.starting_rate
                    correlation_rate = self.args.starting_correlation_rate
                else:
                    rate = self.args.default_rate
                    correlation_rate = self.args.default_correlation_rate

                self.dis_loss_A, self.dis_loss_B, self.gen_loss_A, self.gen_loss_B, self.recon_loss_A, self.recon_loss_B = self._calc_loss(
                    self.generator_A, self.generator_B, self.discriminator_A, self.discriminator_B, A, B, rate)

                self.dis_loss_A_2, self.dis_loss_B_2, self.gen_loss_A_2, self.gen_loss_B_2, self.recon_loss_A_2, self.recon_loss_B_2 = self._calc_loss(self.generator_A_G2, self.generator_B_G2, self.discriminator_A_G2, self.discriminator_B_G2, A, B, rate)

                # correlation loss
                correlation_loss_AB, correlation_loss_BA = self._calc_corr_loss(A, B)

                self.gen_loss_A_2 += correlation_loss_AB * correlation_rate
                self.gen_loss_B_2 += correlation_loss_BA * correlation_rate

                self.gen_loss_1 = self.gen_loss_A + self.gen_loss_B
                self.dis_loss_1 = self.dis_loss_A + self.dis_loss_B
                self.gen_loss_2 = self.gen_loss_A_2 + self.gen_loss_B_2
                self.dis_loss_2 = self.dis_loss_A_2 + self.dis_loss_B_2

                # optimize
                if self.iters % self.args.update_interval == 0:
                    if not self.args.fixed_g1:
                        self.dis_loss_1.backward()
                        self.optim_dis.step()
                    self.dis_loss_2.backward()
                    self.optim_dis_G2.step()
                else:
                    if not self.args.fixed_g1:
                        self.gen_loss_1.backward()
                        self.optim_gen.step()
                    self.gen_loss_2.backward()
                    self.optim_gen_G2.step()

                # log
                self._log_state(A_paths, B_paths, data_A_val, data_B_val)

                self.iters += 1
