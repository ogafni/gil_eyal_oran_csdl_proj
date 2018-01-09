import math
import os
import torch.nn as nn
from torch import optim
from progressbar import ETA, Bar, Percentage, ProgressBar
from itertools import chain
from tensorboardX import SummaryWriter

from .error_bound_calc_functions import calc_mean_gt_error
from .utils import *
from .model import Discriminator, Generator
from .dataset import get_data_loaders


def get_generators(cuda, num_layers, learning_rate):
    generator_A = Generator(num_layers=num_layers)
    generator_B = Generator(num_layers=num_layers)

    if cuda:
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    optim_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
    return generator_A, generator_B, optim_gen


def get_discriminators(cuda, learning_rate):
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    if cuda:
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())
    optim_dis = optim.Adam(dis_params, lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
    return discriminator_A, discriminator_B, optim_dis


def save_images_list(images, suffix, version, board_writer):
    for i, image in enumerate(images):
        name = '{0}_{1}'.format(i, suffix)
        board_writer.add_image(name, image, version)


class DiscoGAN():
    def __init__(self, options):
        self.args = options
        self.cuda = self.args.cuda == 'true'

        self.result_path = self.args.result_path
        self.model_path = self.args.model_path
        self.board_writer = SummaryWriter(os.path.join(self.result_path, "log"))

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        lr = self.args.learning_rate

        if self.args.start_from_pretrained_g1:
            gen_A_path = os.path.join(self.args.pretrained_g1_path_A,
                                      'model_gen_A_G1-' + str(self.args.which_epoch_load))
            self.generator_A = torch.load(gen_A_path)
            gen_B_path = os.path.join(self.args.pretrained_g1_path_B,
                                      'model_gen_B_G1-' + str(self.args.which_epoch_load))
            self.generator_B = torch.load(gen_B_path)
            dis_A_path = os.path.join(self.args.pretrained_g1_path_A,
                                      'model_dis_A_G1-' + str(self.args.which_epoch_load))
            self.discriminator_A = torch.load(dis_A_path)
            dis_B_path = os.path.join(self.args.pretrained_g1_path_B,
                                      'model_dis_B_G1-' + str(self.args.which_epoch_load))
            self.discriminator_B = torch.load(dis_B_path)
            _, _, self.optim_gen = get_generators(self.cuda, self.args.num_layers, lr)
            _, _, self.optim_dis = get_discriminators(self.cuda, lr)
        else:
            self.generator_A, self.generator_B, self.optim_gen = get_generators(self.cuda, self.args.num_layers, lr)
            self.discriminator_A, self.discriminator_B, self.optim_dis = get_discriminators(self.cuda, lr)

        self.recon_criterion = nn.MSELoss()
        self.gan_criterion = nn.BCELoss()
        self.feat_criterion = nn.HingeEmbeddingLoss()

        self.first_image_write = True

    def _save_model(self):
        version = str(int(self.iters / self.args.model_save_interval))
        torch.save(self.generator_A, os.path.join(self.model_path, 'model_gen_A_G1-' + version))
        torch.save(self.generator_B, os.path.join(self.model_path, 'model_gen_B_G1-' + version))
        torch.save(self.discriminator_A, os.path.join(self.model_path, 'model_dis_A_G1-' + version))
        torch.save(self.discriminator_B, os.path.join(self.model_path, 'model_dis_B_G1-' + version))

    def _save_images(self, A, B):
        A, B = read_images(A, B, self.args.image_size, self.cuda)
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
        A, B = read_images(A, B, self.args.image_size, self.cuda, aligned=True)
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
        n_batches = math.ceil(len(data_A) / self.args.batch_size)
        print('%d batches per epoch' % n_batches)
        self.iters = 0
        a_dataloader, b_dataloader = get_data_loaders(data_A, data_B, self.args.batch_size, b_weights)

        for epoch in range(self.args.epoch_size):
            widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=n_batches, widgets=widgets)
            pbar.start()

            for i, (A_paths, B_paths) in enumerate(zip(a_dataloader, b_dataloader)):
                pbar.update(i)

                # read batch data
                A, B = read_images(A_paths, B_paths, self.args.image_size, self.cuda)

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
                self._log_state(A_paths, B_paths, data_A_val, data_B_val)

                self.iters += 1
