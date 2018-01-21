import copy
import numpy as np

from .disco_gan_with_risk import DiscoGANRisk
from .error_bound_calc_functions import samples_order_by_loss_from_filenames


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

    def train(self, data_A, data_B, data_A_val, data_B_val):
        weights = np.ones(len(data_B))
        round = 1
        while not self.stopping_criteria(round, self.bounds, weights):
            print('Round {0}: {1} samples'.format(round, sum(weights > 0)))
            current_model = self._get_round_model(round)
            current_model.train(data_A, data_B, data_A_val, data_B_val, weights)
            G1, G2 = current_model.generator_B, current_model.generator_B_G2
            _, bounds, _ = samples_order_by_loss_from_filenames(data_B, data_B, G1, G2, self.options.cuda,
                                                                self.options.batch_size)
            self.bounds.append(bounds)
            weights = self.weighter(self.bounds, weights)
            round += 1
