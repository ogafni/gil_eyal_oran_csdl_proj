import copy
import numpy as np
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

from .disco_gan_with_risk import DiscoGANRisk
from .error_bound_calc_functions import samples_order_by_loss_from_filenames
from .disco_gan_model import DiscoGAN
from .models_repository import ModelsRepository, _get_last_version

def load_and_print(path):
    print('Loading {}'.format(path))
    return torch.load(path)


def visualize(ref_mat_eb, ref_mat_gt, all_boost_mat_eb, all_boost_mat_gt, final_boost_mat_eb, final_boost_mat_gt):
    n_rounds, n_data = np.shape(all_boost_mat_eb)[0], np.shape(all_boost_mat_eb)[1]
    error_bound = np.append(np.append(ref_mat_eb, np.reshape(all_boost_mat_eb, (n_rounds * n_data))),
                            final_boost_mat_eb)
    ground_truth = np.append(np.append(ref_mat_gt, np.reshape(all_boost_mat_gt, (n_rounds * n_data))),
                             final_boost_mat_gt)
    error = np.append(error_bound, ground_truth)
    iteration = ['reference'] * n_data
    for idx in range(n_rounds):
        iteration.extend([str(idx + 1)] * n_data)
    iteration.extend(['final_boosting'] * n_data)
    iteration.extend(iteration)
    error_type = ['Error Bound'] * int(len(iteration) / 2)
    error_type.extend(['Ground Truth'] * int(len(iteration) / 2))
    data = np.vstack((error, np.vstack((error_type, iteration)))).T
    df = pd.DataFrame(data=data, columns=['Error', 'Error_type', 'Iteration'])
    df['Error'] = pd.to_numeric(df['Error'], errors='ignore')
    gt_error_means = np.append(np.append(np.mean(ref_mat_gt), np.mean(all_boost_mat_gt, axis=1)),
                               np.mean(final_boost_mat_gt))
    eb_error_means = np.append(np.append(np.mean(ref_mat_eb), np.mean(all_boost_mat_eb, axis=1)),
                               np.mean(final_boost_mat_eb))
    sns.set(color_codes=True)
    sns.violinplot(x='Iteration', y='Error', hue='Error_type',
                   data=df, split=True).set_title(
        'Error Bound vs. Ground Truth Error per Iteration', weight='bold')
    plt.plot(np.arange(n_rounds + 2), gt_error_means, 'g', label='Mean Ground Truth Error')
    plt.plot(np.arange(n_rounds + 2), gt_error_means, '.k')
    plt.plot(np.arange(n_rounds + 2), eb_error_means, 'b', label='Mean Error Bound')
    plt.plot(np.arange(n_rounds + 2), eb_error_means, '.k')
    plt.legend()
    plt.show(block=True)

    ## Scatter plot (WIP)
    # g = sns.jointplot(x=df[df['Error_type'] == 'Ground Truth' and df['Iteration'] == '0', 0],
    #                   y=df[df['Error_type'] == 'Ground Truth' and df['Iteration'] == '1', 0],
    #                   kind="kde", color=None)
    # g.plot_joint(plt.scatter, c="b", s=30, linewidth=1, marker="+", label='Ground Truth Error')
    # plt.legend()
    # plt.show(block=True)


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
        self.round_1_model_path = self._get_round_options(1).get_model_path()
        self.bounds = []

    def _get_round_options(self, round):
        round_options = copy.copy(self.options)
        round_options.round = round
        return round_options

    def _get_round_disco_gan(self, round):
        round_options = self._get_round_options(round)
        if round > 1:
            round_options.pretrained_g1 = self.round_1_model_path
        return DiscoGAN(round_options)

    def _get_round_disco_gan_with_risk(self, round):
        round_options = self._get_round_options(round)
        round_options.fixed_g1 = True
        if round > 1:
            round_options.pretrained_g2 = self.round_1_model_path
        return DiscoGANRisk(round_options)

    def _get_boosted_gen(self, round):
        path_g1 = os.path.join(self.options.model_path, self.options.task_name, self.options.dataset, str(round), 'G1')
        path_g2 = os.path.join(self.options.model_path, self.options.task_name, self.options.dataset, str(round), 'G2')
        ver_g1 = _get_last_version(path_g1)
        ver_g2 = _get_last_version(path_g2)
        if self.options.direction_btoa:
            G1 = load_and_print(os.path.join(path_g1, 'gen_A-' + str(ver_g1)))
            G2 = load_and_print(os.path.join(path_g2, 'gen_A-' + str(ver_g2)))
        else:
            G1 = load_and_print(os.path.join(path_g1, 'gen_B-' + str(ver_g1)))
            G2 = load_and_print(os.path.join(path_g2, 'gen_B-' + str(ver_g2)))
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
            print('Training G2')
            gan_with_risk = self._get_round_disco_gan_with_risk(round)
            gan_with_risk.train(data_A, data_B, data_A_val, data_B_val, weights)
            print('Calculating bounds')
            if self.options.direction_btoa:
                G1, G2 = gan_with_risk.generator_A, gan_with_risk.generator_A_G2
                _, bounds, _ = samples_order_by_loss_from_filenames(data_A, data_B, G1, G2, self.options.cuda,
                                                                    self.options.batch_size)
            else:
                G1, G2 = gan_with_risk.generator_B, gan_with_risk.generator_B_G2
                _, bounds, _ = samples_order_by_loss_from_filenames(data_B, data_A, G1, G2, self.options.cuda,
                                                                    self.options.batch_size)
            self.bounds.append(bounds)
            weights = self.weighter(self.bounds, weights)
            round += 1

    def infer(self, data_A_val, data_B_val, method=0):
        """
        :param method: METHODS NOT IMPLEMENTED YET (only 0 is enabled)
        	0 - Use G1 with minimum bound
    		1 - Use G1 with minimum normalized bound
	    	2 - Use G1 with minimum ranking
		    3 - Use last G1 (SelfieBoost)
		    4 - Take the best 1/n on G1_1 and recursively continue
        """
        n_rounds = len(self.bounds)
        repos_model = ModelsRepository(self.options.model_path)
        J_loss_order_boost, J_loss_val_boost, ground_truth_loss_boost = np.zeros((n_rounds, len(data_B_val)),
                                                                                 dtype=int),\
                                                                        np.zeros((n_rounds, len(data_B_val)),
                                                                                 dtype=float), \
                                                                        np.zeros((n_rounds, len(data_B_val)),
                                                                                 dtype=float)  # initialize matrices
        # reference generators (no boosting)
        if self.options.direction_btoa:
            G1_n, _, _, _, _ = repos_model.get_models(G1=True, path=self.options.pretrained_g1)
            G2_n, _, _, _, _ = repos_model.get_models(G1=False, path=self.options.pretrained_g2)
        else:
            _, G1_n, _, _, _ = repos_model.get_models(G1=True, path=self.options.pretrained_g1)
            _, G2_n, _, _, _ = repos_model.get_models(G1=False, path=self.options.pretrained_g2)
        if self.options.cuda:
            G1_n = G1_n.cuda()
            G2_n = G2_n.cuda()
        # val_ref is the error bound for the reference model
        _, J_loss_val_ref, ground_truth_loss_ref = samples_order_by_loss_from_filenames(data_A_val,
                                                                                        data_B_val,
                                                                                        G1_n, G2_n,
                                                                                       self.options.cuda,
                                                                                       self.options.batch_size)
        ref_gt_mean = np.mean(ground_truth_loss_ref)
        for idx_round in range(n_rounds): # run over boosting iterations to find the minimum bound per sample
            G1_n, G2_n = self._get_boosted_gen(idx_round+1) # load generators
            _, J_loss_val_boost[idx_round, :], ground_truth_loss_boost[idx_round, :] = \
                samples_order_by_loss_from_filenames(data_A_val, data_B_val, G1_n, G2_n, self.options.cuda,
                                                     self.options.batch_size) # calculate error bound & gt loss
        min_round_idx, min_sample_idx = np.where(J_loss_val_boost == np.min(J_loss_val_boost, axis=0)) # find min round per sample
        sorted_round_index = np.array([y for x, y in sorted(zip(min_sample_idx, min_round_idx))])  # sort to get a vector of min rounds, representing samples
        final_boost_bound, final_boost_gt = J_loss_val_boost[sorted_round_index, range(len(data_B_val))], \
                                            ground_truth_loss_boost[sorted_round_index, range(len(data_B_val))]  # final generator (predicting min error bound per sample)

        final_boost_gt_mean = np.mean(final_boost_gt)
        print('G0 Ref GT Loss: {:.3}, Boosting GT Loss: {:.3}'.format(np.mean(ref_gt_mean), np.mean(final_boost_gt_mean)))

        visualize(J_loss_val_ref, ground_truth_loss_ref, J_loss_val_boost, ground_truth_loss_boost, final_boost_bound,
                  final_boost_gt) # violin plot (& scatter - still WIP)


        # #min per iteration - WILL COMMIT THIS LATER WITH IN A FUNCTION (& GENERIC)
        #
        # boost_bound_1, boost_gt_1 = J_loss_val_boost[0, :], ground_truth_loss_boost[0, :]
        # min_round_idx_2, min_sample_idx_2 = np.where(J_loss_val_boost[:2, :] == np.min(J_loss_val_boost[:2, :], axis=0))
        # sorted_round_index_2 = np.array([y for x, y in sorted(zip(min_sample_idx_2, min_round_idx_2))])
        # boost_bound_2, boost_gt_2 = J_loss_val_boost[sorted_round_index_2, range(len(data_B_val))], \
        #                             ground_truth_loss_boost[sorted_round_index_2, range(len(data_B_val))]
        # min_round_idx_3, min_sample_idx_3 = np.where(J_loss_val_boost[:3, :] == np.min(J_loss_val_boost[:3, :], axis=0))
        # sorted_round_index_3 = np.array([y for x, y in sorted(zip(min_sample_idx_3, min_round_idx_3))])
        # boost_bound_3, boost_gt_3 = J_loss_val_boost[sorted_round_index_3, range(len(data_B_val))], \
        #                             ground_truth_loss_boost[sorted_round_index_3, range(len(data_B_val))]
        #
        # boost_bound_min_123 =np.vstack((np.vstack((boost_bound_1, boost_bound_2)), boost_bound_3))
        # boost_gt_min_123 = np.vstack((np.vstack((boost_gt_1, boost_gt_2)), boost_gt_3))
        # visualize(J_loss_val_ref, ground_truth_loss_ref, boost_bound_min_123, boost_gt_min_123, final_boost_bound,
        #           final_boost_gt)
        # print('G0 Ref GT Loss: {:.3}, Boosting_0 GT Loss: {:.3}, Boosting_1 GT Loss: {:.3}, Boosting_2 GT Loss: {:.3}, '.format(ref_gt_mean, np.mean(boost_gt_min_123[0, :]), np.mean(boost_gt_min_123[1, :]),np.mean(boost_gt_min_123[2, :])))
        # print('G0 Ref EB Loss: {:.3}, Boosting_0 EB Loss: {:.3}, Boosting_1 EB Loss: {:.3}, Boosting_2 EB Loss: {:.3}, '.format(np.mean(ground_truth_loss_ref), np.mean(boost_bound_min_123[0, :]), np.mean(boost_bound_min_123[1, :]),np.mean(boost_bound_min_123[2, :])))
        #
        # pearson_errors = np.zeros((n_rounds+1))
        # for idx in range(n_rounds):
        #     pearson_errors[idx] = pearsonr(J_loss_val_boost[idx, :], ground_truth_loss_boost[idx, :])[0]
        # pearson_errors[n_rounds] = pearsonr(final_boost_bound, final_boost_gt)[0]
        # print('Correlation between boosted bounds & errors: ', pearson_errors)
        #
        # min_pearson_errors = np.zeros((n_rounds + 1))
        # for idx in range(n_rounds):
        #     min_pearson_errors[idx] = pearsonr(boost_bound_min_123[idx, :], boost_gt_min_123[idx, :])[0]
        #     min_pearson_errors[n_rounds] = pearsonr(final_boost_bound, final_boost_gt)[0]
        # print('Correlation between MIN boosted bounds & errors: ', min_pearson_errors)