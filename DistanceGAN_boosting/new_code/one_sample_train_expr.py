from new_code import dataset
from new_code.options import Options
from new_code.disco_gan_with_risk import DiscoGANRisk
from new_code.disco_gan_model import DiscoGAN
from new_code.error_bound_calc_functions import *
import random
import sys
import numpy as np
import scipy.stats
import seaborn as sns

def train_multiple_g2(data_a, data_b, data_a_val, data_b_val, n_rep=5, n_model_phase=3):
    random.seed(2222)
    batch_size = 64
    error_bound = np.zeros((len(data_a_val),  n_rep))
    ground_truth = np.zeros((len(data_a_val), n_rep))

    n_epochs = int(np.ceil(n_model_phase*10000 / (len(data_a)/batch_size)) + 2)

    rand_idx_vec = random.sample(range(100), n_rep-1)
    rand_idx_vec.insert(0, 0)  # always start with the index 0
    print('Random indices selected: ', rand_idx_vec)
    options_g1 = Options(task_name='one_samp_train', dataset='maps', epoch_size=n_epochs, num_layers=3,
                         num_layers_second_gan=3, continue_training=False, is_auto_detect_training_version=True,
                         model_path='./saved_models')
    model_g1 = DiscoGAN(options_g1)
    model_g1.train(data_a, data_b, data_a_val, data_b_val)
    print('Done training G1')

    for rep_idx in range(n_rep):
        options_g2 = Options(task_name='one_samp_train', dataset='maps', epoch_size=n_epochs,
                             num_layers=3, num_layers_second_gan=3, one_sample_index=rand_idx_vec[rep_idx],
                             continue_training=False, is_auto_detect_training_version=False, fixed_g1=True,
                             which_epoch_load=3, start_from_pretrained_g1=True, batch_size=batch_size,
                             model_path='./saved_models', one_sample_train=True)

        model_g2 = DiscoGANRisk(options_g2)
        model_g2.train(data_a, data_b, data_a_val, data_b_val)
        print('Done training G2 number {} out of {}'.format(rep_idx+1, n_rep))

        _, error_bound[:, rep_idx], ground_truth[:, rep_idx] = samples_order_by_loss_from_filenames(data_a_val, data_b_val,
                                                                                                    model_g1.generator_A,
                                                                                                    model_g2.generator_A_G2,
                                                                                                    options_g2.cuda,
                                                                                                    options_g2.batch_size)
        corr_coeff, p_val = scipy.stats.pearsonr(error_bound[:, rep_idx], ground_truth[:, rep_idx])
        print('{} (sample {}) Correlation Coefficient (P-value): {:.3} ({:.3})'.format(rep_idx, rand_idx_vec[rep_idx],
                                                                                       corr_coeff, p_val))
    return error_bound, ground_truth


def plot_multiple_corr(e_b, g_t):
    color_maps = ['Blues', 'Greens', 'Greys', 'Oranges', 'Purples', 'Reds']
    n_plots = np.shape(e_b)[1]
    for idx in range(n_plots):
        g = sns.kdeplot(e_b[:, idx] + idx*0.1, g_t[:, idx], cmap=color_maps[idx], shade=True, shade_lowest=False, label=idx)
        g.set(xlabel='Error bound (+0.1*idx)', ylabel='Ground truth', title='Error bound vs. Ground truth correlation vs. one_sample_index')

#if __name__ == "__main__":
sys.path.insert(0, 'C:\projects\DLCourse\DistanceGAN_boosting')
data_a, data_b = dataset.get_maps()
data_a_val, data_b_val = dataset.get_maps(test=True)
error_bound, ground_truth = train_multiple_g2(data_a, data_b, data_a_val, data_b_val, n_rep=5, n_model_phase=3)
plot_multiple_corr(error_bound, ground_truth)
print('Stop here')