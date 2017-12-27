from discogan_arch_options.options import Options
from discogan_with_risk_model_g1_fixed import Disco_with_riskGAN
from error_bound_calc_functions import *
import numpy as np
import torch
import os
from operator import itemgetter

options1 = Options(task_name='edges2shoes', model_arch='discogan', result_path='./results/temp/',
                  model_path='./models/temp/',
                  pretrained_g1_path_A='../saved_models/discogan_shoes2edges_epochs98_g2_with_fixed_g1_all_examples/',
                  pretrained_g1_path_B='../saved_models/discogan_shoes2edges_epochs98_g2_with_fixed_g1_all_examples/',
                  pretrained_g2_path_A='../saved_models/discogan_shoes2edges_epochs98_g2_with_fixed_g1_one_sample/',
                  pretrained_g2_path_B='../saved_models/discogan_shoes2edges_epochs98_g2_with_fixed_g1_one_sample/',
                  which_epoch_load=3,batch_size=32, test_mode=True)

options2 = Options(task_name='edges2shoes', model_arch='discogan', result_path='./results/temp/',
                  model_path='./models/temp/',
                   pretrained_g1_path_A='../saved_models/discogan_shoes2edges_epochs98/',
                   pretrained_g1_path_B='../saved_models/discogan_shoes2edges_epochs98/',
                   pretrained_g2_path_A='../saved_models/discogan_shoes2edges_epochs98_g2_with_fixed_g1_one_sample/',
                   pretrained_g2_path_B='../saved_models/discogan_shoes2edges_epochs98_g2_with_fixed_g1_one_sample/',
                  which_epoch_load=3,batch_size=32, test_mode=True)


def load_model(filename):
    model = torch.load(filename)
    model = model.cuda(0)
    model.train(False)
    return model

def load_models(options):

    generator_A_1 = load_model(os.path.join(options.pretrained_g1_path_A, 'model_gen_A-' + str(options.which_epoch_load)))
    generator_B_1 = load_model(os.path.join(options.pretrained_g1_path_B, 'model_gen_B-' + str(options.which_epoch_load)))
    generator_A_2 = load_model(os.path.join(options.pretrained_g2_path_A, 'model_gen_A-' + str(options.which_epoch_load)))
    generator_B_2 = load_model(os.path.join(options.pretrained_g2_path_B, 'model_gen_B-' + str(options.which_epoch_load)))

    return generator_A_1, generator_B_1, generator_A_2, generator_B_2


def run_test_with_boosting(options1, options2, thresh=None):
    model_discogan_with_risk1 = Disco_with_riskGAN(options1)
    model_discogan_with_risk2 = Disco_with_riskGAN(options2)


    G1_0_A, G1_0_B, G2_0_A, G2_0_B = load_models(options1)
    G1_1_A, G1_1_B, G2_1_A, G2_1_B = load_models(options2)


    data_style_A, data_style_B, test_style_A, test_style_B = model_discogan_with_risk1.get_data()


    J_loss_order, J_loss_val, groud_truth_loss = samples_order_by_loss_from_filenames(test_style_A, test_style_B, G1_0_A, G2_0_A, G1_0_B, G2_0_B, options1,
                                         n_batch=64, print_freq=100)

    if thresh is None:
        thresh = np.median(J_loss_val)

    first_round_idx = np.where(J_loss_val <= thresh)
    second_round_idx = np.where(J_loss_val > thresh)

    test_style_A = list(itemgetter(*second_round_idx[0].tolist())(test_style_A))
    test_style_B = list(itemgetter(*second_round_idx[0].tolist())(test_style_B))

    J_loss_order2, J_loss_val2, groud_truth_loss2 = samples_order_by_loss_from_filenames(test_style_A, test_style_B,
                                                                                      G1_1_A, G2_1_A, G1_1_B, G2_1_B,
                                                                                      options2,
                                                                                      n_batch=64, print_freq=100)

    original_ground_truth_loss = np.average(groud_truth_loss)
    boosting_ground_truth_loss = np.average(np.concatenate((groud_truth_loss[first_round_idx],groud_truth_loss2)))

    print("Original Loss: {} Boosting Loss: {}".format(original_ground_truth_loss, boosting_ground_truth_loss))



if __name__ == "__main__":
    run_test_with_boosting(options1, options2)




