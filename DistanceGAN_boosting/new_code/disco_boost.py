import copy
import numpy as np
import torch
import os

from .disco_gan_with_risk import DiscoGANRisk
from .error_bound_calc_functions import samples_order_by_loss_from_filenames
from .disco_gan_model import load_and_print, get_generators

# stopping criterias
class RoundsSC:
    def __init__(self, round):
        self.round = round

    def __call__(self, round, bounds, weights):
        return round >= self.round


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

    def _get_round_model(self, round):
        round_options = copy.copy(self.options)
        round_options.round = round
        return DiscoGANRisk(round_options)

    # def _save_model_round(self, round, G1, G2):
    #     torch.save(G1, os.path.join(self.options.model_path, 'boosted', 'model_gen_A_G1-' + str(round)))
    #     torch.save(G2, os.path.join(self.options.model_path, 'boosted', 'model_gen_A_G2-' + str(round)))

    def _get_boosted_gen(self, round):
        #G1 = load_and_print(os.path.join(self.options.model_path, 'boosted', 'model_gen_A_G1-' + str(round)))
        #G2 = load_and_print(os.path.join(self.options.model_path, 'boosted', 'model_gen_A_G2-' + str(round)))
        G1 = load_and_print(os.path.join(self.options.model_path, 'G1', str(round), 'model_gen_A_G1-'))
        G2 = load_and_print(os.path.join(self.options.model_path, 'G2', str(round), 'model_gen_A_G2-'))
        if self.options.cuda:
            G1 = G1.cuda()
            G2 = G2.cuda()
        return G1, G2

    def train(self, data_A, data_B, data_A_val, data_B_val):
        weights = np.ones(len(data_B))
        round = 1
        while not self.stopping_criteria(round, self.bounds, weights):
            print('Round {0}: {1} samples'.format(round, sum(weights > 0)))
            current_model = self._get_round_model(round)
            current_model.train(data_A, data_B, data_A_val, data_B_val, weights)
            G1, G2 = current_model.generator_A, current_model.generator_A_G2
            _, bounds, _ = samples_order_by_loss_from_filenames(data_B, data_B, G1, G2, self.options.cuda,
                                                                self.options.batch_size)

            self.bounds.append(bounds)
            weights = self.weighter(self.bounds, weights)
            round += 1

    def infer(self, data_A_val, data_B_val):
        n_rounds = len(self.bounds)

        J_loss_order_boost, J_loss_val_boost, ground_truth_loss_boost = np.zeros((len(n_rounds), len(data_B_val)),
                                                                                dtype=int), \
                                                                       np.zeros((len(n_rounds), len(data_B_val)),
                                                                                dtype=float), \
                                                                       np.zeros((len(n_rounds), len(data_B_val)),
                                                                                dtype=float)
        # reference generators (no boosting)
        G1_n, G2_n = get_generators(self.cuda, self.num_layers, self.learning_rate, gen_A_path=self.model_path,
                                    gen_B_path=self.model_path)
        J_loss_order_0, J_loss_val_0, ground_truth_loss_0 = samples_order_by_loss_from_filenames(data_A_val, data_B_val,
                                                                                                G1_n, G2_n, self.cuda,
                                                                                                self.batch_size)
        ref_gt_mean = np.mean(ground_truth_loss_0)
        for idx_round in range(n_rounds):
            G1_n, G2_n = self._get_boosted_gen(idx_round)
            J_loss_order_boost[idx_round, :], J_loss_val_boost[idx_round, :], ground_truth_loss_boost[idx_round, :] = \
                samples_order_by_loss_from_filenames(data_A_val, data_B_val, G1_n, G2_n, self.cuda, self.batch_size)
        min_round_idx, min_sample_idx = np.where(J_loss_val_boost == np.min(J_loss_val_boost, axis=0))

        sorted_round_index = [y for x, y in sorted(zip(min_sample_idx, min_round_idx))]

        final_boost_bound, final_boost_gt = J_loss_val_boost[sorted_round_index, range(len(data_B_val))], \
                                            ground_truth_loss_boost[sorted_round_index, range(len(data_B_val))]

        final_boost_gt_mean = np.mean(final_boost_gt)
        print("G0 Ref GT Loss: {:.3}, Boosting GT Loss: {:.3}".format(np.mean(ref_gt_mean), np.mean(final_boost_gt_mean)))




