import math
import os
import torch.nn as nn
from torch import optim
from progressbar import ETA, Bar, Percentage, ProgressBar
from itertools import chain
import scipy
from tensorboardX import SummaryWriter

from .error_bound_calc_functions import calc_mean_gt_error
from .utils import *
from .model import Discriminator, Generator
from .dataset import shuffle_data


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


def save_images(A, B, G_A, G_B, version, result_path):
    AB = G_B(A)
    BA = G_A(B)
    ABA = G_A(AB)
    BAB = G_B(BA)
    n_testset = min(A.size()[0], B.size()[0])

    subdir_path = os.path.join(result_path, version)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
    for im_idx in range(n_testset):
        A_val = A[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
        B_val = B[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
        BA_val = BA[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
        ABA_val = ABA[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
        AB_val = AB[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
        BAB_val = BAB[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.

        filename_prefix = os.path.join(subdir_path, str(im_idx))
        scipy.misc.imsave(filename_prefix + '.A.jpg', A_val.astype(np.uint8)[:, :, ::-1])
        scipy.misc.imsave(filename_prefix + '.B.jpg', B_val.astype(np.uint8)[:, :, ::-1])
        scipy.misc.imsave(filename_prefix + '.BA.jpg', BA_val.astype(np.uint8)[:, :, ::-1])
        scipy.misc.imsave(filename_prefix + '.AB.jpg', AB_val.astype(np.uint8)[:, :, ::-1])
        scipy.misc.imsave(filename_prefix + '.ABA.jpg', ABA_val.astype(np.uint8)[:, :, ::-1])
        scipy.misc.imsave(filename_prefix + '.BAB.jpg', BAB_val.astype(np.uint8)[:, :, ::-1])


class DiscoGAN():
    def __init__(self, options):
        self.args = options
        self.cuda = self.args.cuda == 'true'

        self.result_path = self.args.result_path
        self.model_path = self.args.model_path
        self.writer_1 = SummaryWriter(os.path.join(self.result_path, "log"))

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        lr = self.args.learning_rate
        self.generator_A, self.generator_B, self.optim_gen = get_generators(self.cuda, self.args.num_layers, lr)
        self.discriminator_A, self.discriminator_B, self.optim_dis = get_discriminators(self.cuda, lr)

        self.recon_criterion = nn.MSELoss()
        self.gan_criterion = nn.BCELoss()
        self.feat_criterion = nn.HingeEmbeddingLoss()

    def _save_model(self):
        version = str(int(self.iters / self.args.model_save_interval))
        torch.save(self.generator_A, os.path.join(self.model_path, 'model_gen_A-' + version))
        torch.save(self.generator_B, os.path.join(self.model_path, 'model_gen_B-' + version))
        torch.save(self.discriminator_A, os.path.join(self.model_path, 'model_dis_A-' + version))
        torch.save(self.discriminator_B, os.path.join(self.model_path, 'model_dis_B-' + version))

    def _save_images(self, A, B):
        A, B = read_images(A, B, self.args.image_size, self.cuda)
        version = str(int(self.iters / self.args.image_save_interval))
        save_images(A, B, self.generator_A, self.generator_B, version, self.result_path)

    def _log_losses(self, A, B):
        A, B = read_images(A, B, self.args.image_size, self.cuda, aligned=True)
        loss = nn.L1Loss()
        gt_error_A = calc_mean_gt_error(B, A, self.generator_A, loss)
        gt_error_B = calc_mean_gt_error(A, B, self.generator_B, loss)
        self.writer_1.add_scalar('GT Error A', gt_error_A, self.iters)
        self.writer_1.add_scalar('GT Error B', gt_error_B, self.iters)

        l2_A = get_model_l2(self.generator_A)
        B_l2 = get_model_l2(self.generator_B)
        self.writer_1.add_scalar('L2 A', l2_A, self.iters)
        self.writer_1.add_scalar('L2 B', B_l2, self.iters)

        self.writer_1.add_scalar('Loss A GEN', as_np(self.gen_loss_A.mean()), self.iters)
        self.writer_1.add_scalar('Loss B GEN', as_np(self.gen_loss_B.mean()), self.iters)
        self.writer_1.add_scalar('Loss A DIS', as_np(self.dis_loss_A.mean()), self.iters)
        self.writer_1.add_scalar('Loss B DIS', as_np(self.dis_loss_B.mean()), self.iters)
        self.writer_1.add_scalar('Loss A RECON', as_np(self.recon_loss_A.mean()), self.iters)
        self.writer_1.add_scalar('Loss B RECON', as_np(self.recon_loss_B.mean()), self.iters)

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

    def train(self, data_A, data_B, data_A_val, data_B_val):
        n_batches = math.ceil(len(data_A) / self.args.batch_size)
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
                if self.iters % self.args.log_interval == 0:
                    self._log_losses(A_paths, B_paths)
                if self.iters % self.args.image_save_interval == 0:
                    self._save_images(data_A_val, data_B_val)
                if self.iters % self.args.model_save_interval == 0:
                    self._save_model()

                self.iters += 1
