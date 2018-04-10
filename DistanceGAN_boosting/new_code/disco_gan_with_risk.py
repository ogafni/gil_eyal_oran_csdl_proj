import math
import os
import torch.nn as nn
from progressbar import ETA, Bar, Percentage, ProgressBar

from .dataset import get_data_loaders
from .disco_gan_model import DiscoGAN, get_xnators
from .error_bound_calc_functions import calc_mean_gt_error, calc_mean_error_bound, calc_correlation
from .utils import *


class DiscoGANRisk(DiscoGAN):
    def __init__(self, options):
        super().__init__(options)

        self.result_path_1 = os.path.join(self.result_path, 'G1')
        self.result_path_2 = os.path.join(self.result_path, 'G2')
        os.makedirs(self.result_path_1, exist_ok=True)
        os.makedirs(self.result_path_2, exist_ok=True)

        if self.args.is_auto_detect_training_version and self.models_repository.has_models(False):
            gen_a, gen_b, dis_a, dis_b, self.last_exist_model_g2 = self.models_repository.get_models(False)
        else:
            self.last_exist_model_g2 = 0
            if self.args.pretrained_g2:
                gen_a, gen_b, dis_a, dis_b, _ = self.models_repository.get_models(False, path=self.args.pretrained_g2,
                                                                                  wanted_version=self.args.which_epoch_load)
            else:
                gen_a, gen_b, dis_a, dis_b = self._get_new_models()

        lr = self.args.learning_rate
        self.generator_A_G2, self.generator_B_G2, self.optim_gen_G2 = get_xnators(gen_a, gen_b, self.cuda, lr)
        self.discriminator_A_G2, self.discriminator_B_G2, self.optim_dis_G2 = get_xnators(dis_a, dis_b, self.cuda, lr)

        self.correlation_criterion = nn.L1Loss()

    def _save_model(self):
        version = str(int(self.iters / self.args.model_save_interval))
        if not self.args.fixed_g1:
            super()._save_model()
        self.models_repository.save_model(self.generator_A_G2, self.generator_B_G2, self.discriminator_A_G2,
                                          self.discriminator_B_G2, version, False)
        if version == str(self.args.version_save):
            self.is_keep_training = False

    def _log_losses(self, A, B):
        if not self.args.fixed_g1:
            super()._log_losses(A, B)
        A, B = read_images(A, B, self.args.image_size, self.cuda, self.args.dataset, aligned=True)
        loss = nn.L1Loss()
        gt_error_A_G2 = calc_mean_gt_error(B, A, self.generator_A_G2, loss)
        gt_error_B_G2 = calc_mean_gt_error(A, B, self.generator_B_G2, loss)
        self.board_writer.add_scalar('GT Error A G2', gt_error_A_G2, self.iters)
        self.board_writer.add_scalar('GT Error B G2', gt_error_B_G2, self.iters)
        error_bound_A = calc_mean_error_bound(B, self.generator_A, self.generator_A_G2, loss)
        error_bound_B = calc_mean_error_bound(A, self.generator_B, self.generator_B_G2, loss)
        self.board_writer.add_scalar('Bound A', error_bound_A, self.iters)
        self.board_writer.add_scalar('Bound B', error_bound_B, self.iters)
        correlation_A = calc_correlation(B, A, self.generator_A, self.generator_A_G2)
        correlation_B = calc_correlation(A, B, self.generator_B, self.generator_B_G2)
        self.board_writer.add_scalar('Correlation A', correlation_A, self.iters)
        self.board_writer.add_scalar('Correlation B', correlation_B, self.iters)

    def _calc_corr_loss(self, A, B):
        AB_1 = self.generator_B(A)
        BA_1 = self.generator_A(B)
        AB_2 = self.generator_B_G2(A)
        BA_2 = self.generator_A_G2(B)
        if self.args.one_sample_train:
            one_sample_index = self.args.one_sample_index
            AB_1, BA_1 = AB_1[one_sample_index], BA_1[one_sample_index]
            AB_2, BA_2 = AB_2[one_sample_index], BA_2[one_sample_index]
        # Correlation loss
        # Distance between generator 1 and generator 2's output
        correlation_loss_AB_2 = - self.correlation_criterion(AB_2, to_no_grad_var(AB_1))
        correlation_loss_BA_2 = - self.correlation_criterion(BA_2, to_no_grad_var(BA_1))
        return correlation_loss_AB_2, correlation_loss_BA_2

    def _calc_loss_one(self, generator_A, generator_B, discriminator_A, discriminator_B, A, B, one_sample_A,
                       one_sample_B, rate):
        one_sample_index = self.args.one_sample_index
        generator_A.zero_grad()
        generator_B.zero_grad()
        discriminator_A.zero_grad()
        discriminator_B.zero_grad()

        AB = generator_B(A)
        BA = generator_A(B)
        ABA = generator_A(AB)
        BAB = generator_B(BA)

        one_sample_AB_full = generator_B(one_sample_A)
        one_sample_BA_full = generator_A(one_sample_B)

        one_sample_ABA = generator_A(one_sample_AB_full)[one_sample_index]
        one_sample_BAB = generator_B(one_sample_BA_full)[one_sample_index]

        # Reconstruction Loss
        recon_loss_A = self.recon_criterion(ABA, A)
        recon_loss_B = self.recon_criterion(BAB, B)
        one_sample_recon_loss_A = self.recon_criterion(one_sample_ABA, one_sample_A[one_sample_index])
        one_sample_recon_loss_B = self.recon_criterion(one_sample_BAB, one_sample_B[one_sample_index])

        # Real/Fake GAN Loss (A)
        A_dis_real, A_feats_real = discriminator_A(A)
        A_dis_fake, A_feats_fake = discriminator_A(BA)
        dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, self.gan_criterion, self.cuda)
        fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, self.feat_criterion)

        # Real/Fake GAN Loss (B)
        B_dis_real, B_feats_real = discriminator_B(B)
        B_dis_fake, B_feats_fake = discriminator_B(AB)
        dis_loss_B, gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, self.gan_criterion, self.cuda)
        fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, self.feat_criterion)

        # Additional for one-sample mode (A)
        A_dis_real, A_feats_real = discriminator_A(one_sample_A[one_sample_index: one_sample_index + 1])
        A_dis_fake, A_feats_fake = discriminator_A(one_sample_BA_full[one_sample_index: one_sample_index + 1])
        one_sample_dis_loss_A, one_sample_gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, self.gan_criterion,
                                                                    self.cuda)
        one_sample_fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, self.feat_criterion)
        gen_loss_A += one_sample_gen_loss_A
        fm_loss_A += one_sample_fm_loss_A
        recon_loss_A += one_sample_recon_loss_A

        # Additional for one-sample mode (A)
        B_dis_real, B_feats_real = discriminator_B(one_sample_B[one_sample_index: one_sample_index + 1])
        B_dis_fake, B_feats_fake = discriminator_B(one_sample_AB_full[one_sample_index: one_sample_index + 1])
        one_sample_dis_loss_B, one_sample_gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, self.gan_criterion,
                                                                    self.cuda)
        one_sample_fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, self.feat_criterion)
        gen_loss_B += one_sample_gen_loss_B
        fm_loss_B += one_sample_fm_loss_B
        recon_loss_B += one_sample_recon_loss_B

        # Total Loss
        gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * (1. - rate) + recon_loss_A * rate
        gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * (1. - rate) + recon_loss_B * rate

        return dis_loss_A, dis_loss_B, gen_loss_A_total, gen_loss_B_total, recon_loss_A, recon_loss_B

    def train(self, data_A, data_B, data_A_val, data_B_val, b_weights=None):

        if self.args.is_auto_detect_training_version and self.last_exist_model_g2 == 3:
            return

        n_batches = math.ceil((len(data_A)) / self.args.batch_size)
        print('%d batches per epoch' % n_batches)
        if self.args.is_auto_detect_training_version:
            self.iters = self.last_exist_model_g2 * self.args.model_save_interval
        else:
            self.iters = 0
        a_dataloader, b_dataloader = get_data_loaders(data_A, data_B, self.args.batch_size, b_weights)

        if self.args.one_sample_train:
            one_sample_A, one_sample_B = read_images(data_A_val[:100], data_A_val[:100],
                                                     self.args.image_size, self.cuda, self.args.dataset)

        for epoch in range(self.args.epoch_size):

            widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=n_batches, widgets=widgets)
            pbar.start()

            for i, (A_paths, B_paths) in enumerate(zip(a_dataloader, b_dataloader)):
                pbar.update(i)

                # read batch data
                A, B = read_images(A_paths, B_paths, self.args.image_size, self.cuda, self.args.dataset)

                # calculate losses
                correlation_rate = self.args.default_correlation_rate
                if self.iters < self.args.gan_curriculum:
                    if not self.args.one_sample_train:
                        correlation_rate = self.args.starting_correlation_rate
                    rate = self.args.starting_rate
                else:
                    rate = self.args.default_rate

                # Calculate G1 for both cases
                self.dis_loss_A, self.dis_loss_B, self.gen_loss_A, self.gen_loss_B, self.recon_loss_A, self.recon_loss_B = \
                    self._calc_loss(self.generator_A, self.generator_B, self.discriminator_A, self.discriminator_B, A,
                                    B, rate)
                # Calculate G2 for one-sample mode
                if self.args.one_sample_train:
                    self.dis_loss_A_2, self.dis_loss_B_2, self.gen_loss_A_2, self.gen_loss_B_2, self.recon_loss_A_2, self.recon_loss_B_2 = \
                        self._calc_loss_one(self.generator_A_G2, self.generator_B_G2, self.discriminator_A_G2,
                                            self.discriminator_B_G2, A, B, one_sample_A, one_sample_B, rate)
                    correlation_loss_AB, correlation_loss_BA = self._calc_corr_loss(one_sample_A, one_sample_B)

                # Calculate G2 for all-samples case
                else:
                    self.dis_loss_A_2, self.dis_loss_B_2, self.gen_loss_A_2, self.gen_loss_B_2, self.recon_loss_A_2, self.recon_loss_B_2 = \
                        self._calc_loss(self.generator_A_G2, self.generator_B_G2, self.discriminator_A_G2,
                                        self.discriminator_B_G2, A, B, rate)
                    correlation_loss_AB, correlation_loss_BA = self._calc_corr_loss(A, B)

                self.gen_loss_A_2 += correlation_loss_AB * correlation_rate
                self.gen_loss_B_2 += correlation_loss_BA * correlation_rate

                if self.args.model_arch == 'discogan':
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

                self.iters += 1

                self._log_state(A_paths, B_paths, data_A_val, data_B_val)

                if not self.is_keep_training:
                    break

            if not self.is_keep_training:
                break
