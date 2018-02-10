import copy
import numpy as np
import os
import torch

from .disco_gan_with_risk import DiscoGANRisk
from .error_bound_calc_functions import samples_order_by_loss_from_filenames
from .disco_gan_model import DiscoGAN


def load_and_print(path):
    print('Loading {}'.format(path))
    return torch.load(path)


# stopping criterias
class RoundsSC:
    def __init__(self, round):
        self.round = round

    def __call__(self, round, bounds, weights):
        return round > self.round


class DataSizeSC:
    def __init__(self, min_samples):
        self.min_samples = min_samples

    def __call__(self, round, bounds, weights):
        num_samples = (weights > 0).sum()
        return self.min_samples > num_samples


# weighters
class KeepBadWeighter:
    def __init__(self, fraction_to_keep):
        self.percentile = (1 - fraction_to_keep) * 100

    def __call__(self, bounds, weights):
        last_bounds = bounds[-1]
        threshold = np.percentile(last_bounds[weights > 0], self.percentile)
        return np.bitwise_and(last_bounds > threshold, weights > 0).astype(int)


class ADAWeighter:
    def __call__(self, bounds, weights):
        return bounds[-1]


class DiscoBoost:
    def __init__(self, options, weighter, stopping_criteria):
        self.options = options
        self.weighter = weighter
        self.stopping_criteria = stopping_criteria
        self.bounds = []

    def _get_round_options(self, round):
        round_options = copy.copy(self.options)
        round_options.round = round
        return round_options

    def _get_round_disco_gan(self, round):
        round_options = self._get_round_options(round)
        return DiscoGAN(round_options)

    def _get_round_disco_gan_with_risk(self, round):
        round_options = self._get_round_options(round)
        round_options.fixed_g1 = True
        return DiscoGANRisk(round_options)


    def _get_boosted_gen(self, round):
        G1 = load_and_print(os.path.join(self.options.model_path, self.options.task_name, self.options.dataset,
                                         str(round), 'G1', 'model_gen_A_G1-' + str(self.options.which_epoch_load)))
        G2 = load_and_print(os.path.join(self.options.model_path, self.options.task_name, self.options.dataset,
                                         str(round), 'G2', 'model_gen_A_G2-' + str(self.options.which_epoch_load)))
        if self.options.cuda:
            G1 = G1.cuda()
            G2 = G2.cuda()
        return G1, G2

    def _get_reference_gen(self, G1_path, G2_path):
        G1 = load_and_print(G1_path)
        G2 = load_and_print(G2_path)
        if self.options.cuda:
            G1 = G1.cuda()
            G2 = G2.cuda()
        return G1, G2

    def train(self, data_A, data_B, data_A_val, data_B_val):
        weights = np.ones(len(data_B))
        round = 1
        while not self.stopping_criteria(round, self.bounds, weights):
            print('Round {0}: {1} samples'.format(round, sum(weights > 0)))
            gan = self._get_round_disco_gan(round)
            gan.train(data_A, data_B, data_A_val, data_B_val, weights)
            gan_with_risk = self._get_round_disco_gan_with_risk(round)
            gan_with_risk.train(data_A, data_B, data_A_val, data_B_val, weights)
            if self.options.direction_btoa:
                G1, G2 = gan_with_risk.generator_A, gan_with_risk.generator_A_G2
                _, bounds, _ = samples_order_by_loss_from_filenames(data_B, data_B, G1, G2, self.options.cuda,
                                                                    self.options.batch_size)
            else:
                G1, G2 = gan_with_risk.generator_B, gan_with_risk.generator_B_G2
                _, bounds, _ = samples_order_by_loss_from_filenames(data_A, data_A, G1, G2, self.options.cuda,
                                                                    self.options.batch_size)
            self.bounds.append(bounds)
            weights = self.weighter(self.bounds, weights)
            round += 1

    def infer(self, data_A_val, data_B_val):
        n_rounds = len(self.bounds)

        J_loss_order_boost, J_loss_val_boost, ground_truth_loss_boost = np.zeros((n_rounds, len(data_B_val)),
                                                                                dtype=int), \
                                                                       np.zeros((n_rounds, len(data_B_val)),
                                                                                dtype=float), \
                                                                       np.zeros((n_rounds, len(data_B_val)),
                                                                                dtype=float)
        # reference generators (no boosting) model_gen_B_G1-3

        if self.options.direction_btoa:
            reference_model_path_G1 = os.path.join('./saved_models/reference', self.options.dataset, 'G1',
                                                   'model_gen_A_G1-' + str(self.options.which_epoch_load))
            reference_model_path_G2 = os.path.join('./saved_models/reference', self.options.dataset, 'G2',
                                                   'model_gen_A_G2-' + str(self.options.which_epoch_load))
        else:
            reference_model_path_G1 = os.path.join('./saved_models/reference', self.options.dataset, 'G1',
                                                   'model_gen_B_G1-' + str(self.options.which_epoch_load))
            reference_model_path_G2 = os.path.join('./saved_models/reference', self.options.dataset, 'G2',
                                                   'model_gen_B_G2-' + str(self.options.which_epoch_load))
        G1_n, G2_n = self._get_reference_gen(reference_model_path_G1, reference_model_path_G2)
        J_loss_order_ref, J_loss_val_ref, ground_truth_loss_ref = samples_order_by_loss_from_filenames(data_A_val,
                                                                                                       data_B_val,
                                                                                                       G1_n, G2_n,
                                                                                                       self.options.cuda,
                                                                                                       self.options.batch_size)
        ref_gt_mean = np.mean(ground_truth_loss_ref)
        for idx_round in range(n_rounds):
            G1_n, G2_n = self._get_boosted_gen(idx_round+1)
            J_loss_order_boost[idx_round, :], J_loss_val_boost[idx_round, :], ground_truth_loss_boost[idx_round, :] = \
                samples_order_by_loss_from_filenames(data_A_val, data_B_val, G1_n, G2_n, self.options.cuda,
                                                     self.options.batch_size)
        min_round_idx, min_sample_idx = np.where(J_loss_val_boost == np.min(J_loss_val_boost, axis=0))

        sorted_round_index = [y for x, y in sorted(zip(min_sample_idx, min_round_idx))]

        final_boost_bound, final_boost_gt = J_loss_val_boost[sorted_round_index, range(len(data_B_val))], \
                                            ground_truth_loss_boost[sorted_round_index, range(len(data_B_val))]

        final_boost_gt_mean = np.mean(final_boost_gt)
        print('G0 Ref GT Loss: {:.3}, Boosting GT Loss: {:.3}'.format(np.mean(ref_gt_mean), np.mean(final_boost_gt_mean)))




