import math
import os
import torch.nn as nn
from torch import optim
from progressbar import ETA, Bar, Percentage, ProgressBar
from itertools import chain
from tensorboardX import SummaryWriter

from .models_repository import ModelsRepository
from .error_bound_calc_functions import calc_mean_gt_error
from .utils import *
from .model import Discriminator, Generator
from .dataset import get_data_loaders


def get_xnators(Xnator_A, Xnator_B, cuda, learning_rate):
    if cuda:
        Xnator_A = Xnator_A.cuda()
        Xnator_B = Xnator_B.cuda()

    dis_params = chain(Xnator_A.parameters(), Xnator_B.parameters())
    optim_dis = optim.Adam(dis_params, lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
    return Xnator_A, Xnator_B, optim_dis


def save_images_list(images, suffix, version, board_writer):
    for i, image in enumerate(images):
        name = '{0}_{1}'.format(i, suffix)
        board_writer.add_image(name, image, version)


class DiscoGAN():
    def __init__(self, options):
        self.args = options
        self.cuda = self.args.cuda == 'true'

        self.result_path = self.args.get_result_path()
        os.makedirs(self.result_path, exist_ok=True)
        self.board_writer = SummaryWriter(os.path.join(self.result_path, "log"))

        self.models_repository = ModelsRepository(self.args.get_model_path())

        self.is_keep_training = True
        self.first_image_write = True

        if self.args.is_auto_detect_training_version and self.models_repository.has_models():
            gen_a, gen_b, dis_a, dis_b, self.last_exist_model_g1 = self.models_repository.get_models()
        else:
            self.last_exist_model_g1 = 0
            if self.args.pretrained_g1:
                gen_a, gen_b, dis_a, dis_b, _ = self.models_repository.get_models(path=self.args.pretrained_g1)
            else:
                gen_a, gen_b, dis_a, dis_b = self._get_new_models()

        lr = self.args.learning_rate
        self.generator_A, self.generator_B, self.optim_gen = get_xnators(gen_a, gen_b, self.cuda, lr)
        self.discriminator_A, self.discriminator_B, self.optim_dis = get_xnators(dis_a, dis_b, self.cuda, lr)

        self.recon_criterion = nn.MSELoss()
        self.gan_criterion = nn.BCELoss()
        self.feat_criterion = nn.HingeEmbeddingLoss()

    def _get_new_models(self):
        gen_a = Generator(num_layers=self.args.num_layers)
        gen_b = Generator(num_layers=self.args.num_layers)
        dis_a = Discriminator()
        dis_b = Discriminator()
        return gen_a, gen_b, dis_a, dis_b

    def _save_model(self):
        version = str(int(self.iters / self.args.model_save_interval))
        self.models_repository.save_model(self.generator_A, self.generator_B, self.discriminator_A,
                                          self.discriminator_B, version)
        if version == 'self.args.version_save':
            self.is_keep_training = False

    def _save_images(self, A, B):
        A, B = read_images(A, B, self.args.image_size, self.cuda, self.args.dataset)
        version = str(int(self.iters / self.args.image_save_interval))
        if self.first_image_write:
            save_images_list(A, 'A', version, self.board_writer)
            save_images_list(B, 'B', version, self.board_writer)
            self.first_image_write = False
        AB = self.generator_B(A)
        save_images_list(AB, 'AB', version, self.board_writer)
        # freeing some precious GPU space
        del AB
        BA = self.generator_A(B)
        save_images_list(BA, 'BA', version, self.board_writer)

    def _log_losses(self, A, B):
        A, B = read_images(A, B, self.args.image_size, self.cuda, self.args.dataset, aligned=True)
        loss = nn.L1Loss()
        gt_error_A = calc_mean_gt_error(B, A, self.generator_A, loss)
        gt_error_B = calc_mean_gt_error(A, B, self.generator_B, loss)
        self.board_writer.add_scalar('GT Error A', gt_error_A, self.iters)
        self.board_writer.add_scalar('GT Error B', gt_error_B, self.iters)

        l2_A = get_model_l2(self.generator_A)
        B_l2 = get_model_l2(self.generator_B)
        self.board_writer.add_scalar('L2 A', l2_A, self.iters)
        self.board_writer.add_scalar('L2 B', B_l2, self.iters)

        self.board_writer.add_scalar('Loss A GEN', as_np(self.gen_loss_A.mean()), self.iters)
        self.board_writer.add_scalar('Loss B GEN', as_np(self.gen_loss_B.mean()), self.iters)
        self.board_writer.add_scalar('Loss A DIS', as_np(self.dis_loss_A.mean()), self.iters)
        self.board_writer.add_scalar('Loss B DIS', as_np(self.dis_loss_B.mean()), self.iters)
        self.board_writer.add_scalar('Loss A RECON', as_np(self.recon_loss_A.mean()), self.iters)
        self.board_writer.add_scalar('Loss B RECON', as_np(self.recon_loss_B.mean()), self.iters)

    def _calc_loss(self, generator_A, generator_B, discriminator_A, discriminator_B, A, B, rate):
        generator_A.zero_grad()
        generator_B.zero_grad()
        discriminator_A.zero_grad()
        discriminator_B.zero_grad()

        AB = generator_B(A)
        BA = generator_A(B)
        ABA = generator_A(AB)
        BAB = generator_B(BA)

        # Reconstruction Loss
        recon_loss_A = self.recon_criterion(ABA, A)
        recon_loss_B = self.recon_criterion(BAB, B)

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

        # Total Loss
        gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * (1. - rate) + recon_loss_A * rate
        gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * (1. - rate) + recon_loss_B * rate

        return dis_loss_A, dis_loss_B, gen_loss_A_total, gen_loss_B_total, recon_loss_A, recon_loss_B

    def _log_state(self, A_paths, B_paths, data_A_val, data_B_val):
        if self.iters % self.args.log_interval == 0:
            self._log_losses(A_paths, B_paths)
        if self.iters % self.args.image_save_interval == 0:
            self._save_images(data_A_val, data_B_val)
        if self.iters % self.args.model_save_interval == 0:
            self._save_model()

    def train(self, data_A, data_B, data_A_val, data_B_val, b_weights=None):

        if self.args.is_auto_detect_training_version and self.last_exist_model_g1 == self.args.which_epoch_load:
            return

        n_batches = math.ceil(len(data_A) / self.args.batch_size)
        print('%d batches per epoch' % n_batches)
        if self.args.is_auto_detect_training_version:
            self.iters = self.last_exist_model_g1 * self.args.model_save_interval
        else:
            self.iters = 0

        a_dataloader, b_dataloader = get_data_loaders(data_A, data_B, self.args.batch_size, b_weights)

        for epoch in range(self.args.epoch_size):
            widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=n_batches, widgets=widgets)
            pbar.start()

            for i, (A_paths, B_paths) in enumerate(zip(a_dataloader, b_dataloader)):
                pbar.update(i)

                # read batch data
                A, B = read_images(A_paths, B_paths, self.args.image_size, self.cuda, self.args.dataset)

                # calculate losses
                if self.iters < self.args.gan_curriculum:
                    rate = self.args.starting_rate
                else:
                    rate = self.args.default_rate

                self.dis_loss_A, self.dis_loss_B, self.gen_loss_A, self.gen_loss_B, self.recon_loss_A, self.recon_loss_B = self._calc_loss(
                    self.generator_A, self.generator_B, self.discriminator_A, self.discriminator_B, A, B, rate)

                self.gen_loss = self.gen_loss_A + self.gen_loss_B
                self.dis_loss = self.dis_loss_A + self.dis_loss_B

                # optimize
                if self.iters % self.args.update_interval == 0:
                    self.dis_loss.backward()
                    self.optim_dis.step()
                else:
                    self.gen_loss.backward()
                    self.optim_gen.step()

                # log
                self.iters += 1

                self._log_state(A_paths, B_paths, data_A_val, data_B_val)

                if not self.is_keep_training:
                    break

            if not self.is_keep_training:
                break
