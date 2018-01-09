import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import os
import matplotlib.pyplot as plt
import scipy.stats
import math
from new_code.options import Options
from new_code import dataset
from new_code.disco_gan_with_risk import DiscoGANRisk
from new_code.disco_gan_model import DiscoGAN
from new_code import utils
from new_code.inference import *
from new_code.error_bound_calc_functions import *

# sys.path.insert(0, 'C:\projects\DLCourse\DistanceGAN_boosting')
# sys.path.insert(0, 'C:\projects\DLCourse\DistanceGAN_boosting\discogan_arch')
sys.path.insert(0, '/home/deepjunior/anaconda3/projects/gil_eyal_oran_csdl_proj/DistanceGAN_boosting')
sys.path.insert(0, '/home/deepjunior/anaconda3/projects/gil_eyal_oran_csdl_proj/DistanceGAN_boosting/discogan_arch')


def initial_boosting_calc(pretrained_g10_path, pretrained_g20_path, data_a, data_b):
    options_fixed = Options(task_name='edges2shoes', model_arch='discogan', which_epoch_load=3,
                            not_all_samples=False, batch_size=64, fixed_g1=True, start_from_pretrained_g1=True,
                            start_from_pretrained_g2=True,
                            pretrained_g1_path_A=pretrained_g10_path, pretrained_g1_path_B=pretrained_g10_path,
                            pretrained_g2_path_A=pretrained_g20_path, pretrained_g2_path_B=pretrained_g20_path)
    #     fixed_disco_risk = DiscoGANRisk(options_fixed)
    G1_0_A, G1_0_B, G2_0_A, G2_0_B = load_models(options_fixed)

    J_loss_order, J_loss_val, groud_truth_loss = samples_order_by_loss_from_filenames(data_a, data_b,
                                                                                      G1_0_B, G2_0_B,
                                                                                      options_fixed.cuda,
                                                                                      options_fixed.batch_size)
    return J_loss_order, J_loss_val, groud_truth_loss


def return_partial_ordered(data_a, data_b, J_loss_order, keep_portion=0.5):
    n_keep = int(np.round(keep_portion * len(J_loss_order)))
    data_a_reorder = reorder_samples_by_loss(J_loss_order, data_a)
    data_b_reorder = reorder_samples_by_loss(J_loss_order, data_b)
    #     data_a_reorder, data_b_reorder = data_a[J_loss_order[:n_keep]], data_b[J_loss_order[:n_keep]]

    return data_a_reorder[:n_keep], data_b_reorder[:n_keep]


def boosting_train(pretrained_g11_path, pretrained_g21_path, data_a_reorder_0, data_b_reorder_0, data_A_val, data_B_val,
                   phase=2, over_train=2, boosting_phases=1, b_weights=None):
    new_g1_path = pretrained_g11_path + '_boosting1_g1'
    new_g2_path = pretrained_g21_path + '_boosting1_g2'
    options_no_risk = Options(task_name='edges2shoes', model_arch='discogan', which_epoch_load=phase,
                              epoch_size=27 * over_train, batch_size=64, start_from_pretrained_g1=True,
                              pretrained_g1_path_A=pretrained_g11_path, pretrained_g1_path_B=pretrained_g11_path,
                              model_path=new_g1_path, fixed_g1=False)
    discogan_model_no_risk = DiscoGAN(options_no_risk)
    print('Starting to train G1_1')
    discogan_model_no_risk.train(data_a_reorder_0, data_b_reorder_0, data_A_val, data_B_val, b_weights=None)
    print('Done training G1_1. Saved at:')
    print(new_g1_path)
    options_with_risk = Options(task_name='edges2shoes', model_arch='discogan',
                                epoch_size=(phase + 1) * 27 * over_train,
                                which_epoch_load=boosting_phases, start_from_pretrained_g1=True, one_sample_train=True,
                                pretrained_g1_path_A=new_g1_path, pretrained_g1_path_B=new_g1_path,
                                fixed_g1=True, batch_size=64, model_path=new_g2_path)
    discogan_model_with_risk = DiscoGANRisk(options_with_risk)

    print('Starting G2_1')
    discogan_model_with_risk.train(data_a_reorder_0, data_b_reorder_0, data_A_val, data_B_val, b_weights=None)
    print('Done training G2_1. Saved at:')
    print(new_g2_path)
    return new_g1_path, new_g2_path


def define_options_for_inference(g1_0_path, g2_0_path, g1_1_path, g2_1_path, phase=2, over_train=2):
    options1 = Options(task_name='edges2shoes', model_arch='discogan', result_path='./results/temp/',
                       model_path='./models/temp/', pretrained_g1_path_A=g1_0_path, pretrained_g1_path_B=g1_0_path,
                       pretrained_g2_path_A=g2_0_path, pretrained_g2_path_B=g2_0_path,
                       which_epoch_load=3, batch_size=64, test_mode=True)

    options2 = Options(task_name='edges2shoes', model_arch='discogan', result_path='./results/temp/',
                       model_path='./models/temp/', pretrained_g1_path_A=g1_1_path, pretrained_g1_path_B=g1_1_path,
                       pretrained_g2_path_A=g2_1_path, pretrained_g2_path_B=g2_1_path,
                       which_epoch_load=3, batch_size=64, test_mode=True)
    return options1, options2


def boosting_train_only_g2_train(pretrained_g11_path, pretrained_g21_path, data_a_reorder_0, data_b_reorder_0,
                                 data_A_val, data_B_val,
                                 phase=2, over_train=2, boosting_phases=1, b_weights=None):
    new_g1_path = pretrained_g11_path + '_boosting1_g1'
    new_g2_path = pretrained_g21_path + '_boosting1_g2'

    options_with_risk = Options(task_name='edges2shoes', model_arch='discogan',
                                epoch_size=(phase + 1) * 27 * over_train,
                                which_epoch_load=boosting_phases, start_from_pretrained_g1=True, one_sample_train=True,
                                pretrained_g1_path_A=new_g1_path, pretrained_g1_path_B=new_g1_path,
                                fixed_g1=True, batch_size=64, model_path=new_g2_path)
    discogan_model_with_risk = DiscoGANRisk(options_with_risk)

    print('Starting G2_1')
    discogan_model_with_risk.train(data_a_reorder_0, data_b_reorder_0, data_A_val, data_B_val, b_weights=None)
    print('Done training G2_1. Saved at:')
    print(new_g2_path)
    return new_g1_path, new_g2_path


def boosting_no_train(pretrained_g11_path, pretrained_g21_path, data_a_reorder_0, data_b_reorder_0, data_A_val,
                      data_B_val,
                      phase=2, over_train=2, boosting_phases=1, b_weights=None):
    new_g1_path = pretrained_g11_path + '_boosting1_g1'
    new_g2_path = pretrained_g21_path + '_boosting1_g2'

    return new_g1_path, new_g2_path


if __name__ == "__main__":
    # Run with G1 & G2 training
    pretrained_g10_path, pretrained_g20_path = './saved_models/discogan_shoes2edges/g1_only', \
                                               './saved_models/discogan_shoes2edges/g2_one-sample_g1-fixed'
    data_a, data_b = dataset.get_edges2shoes()
    data_len = min(len(data_a), len(data_b))
    data_a, data_b = data_a[:data_len], data_b[:data_len]

    data_A_val, data_B_val = dataset.get_edges2shoes(test=True)
    J_loss_order_0, J_loss_val_0, ground_truth_loss_0 = initial_boosting_calc(pretrained_g10_path, pretrained_g20_path,
                                                                              data_a=data_a, data_b=data_b)
    data_a_reorder_0, data_b_reorder_0 = return_partial_ordered(data_a, data_b, J_loss_order_0, keep_portion=0.5)
    over_train = int(round(1 / (len(data_a_reorder_0) / len(data_a))))
    g1_1_path, g2_1_path = boosting_train(pretrained_g10_path, pretrained_g20_path, data_a_reorder_0, data_b_reorder_0,
                                          data_A_val, data_B_val, phase=2, over_train=over_train, boosting_phases=1,
                                          b_weights=None)
    ### STILL NEED TO DEFINE A BETTER WAY TO LOAD THE MODEL PHASE
    options_0, options_1 = define_options_for_inference(pretrained_g10_path, pretrained_g20_path,
                                                        pretrained_g10_path + '_boosting1_g1',
                                                        pretrained_g20_path + '_boosting1_g2', phase=2, over_train=2)
    gt_loss_g0, gt_loss_boosting, gt_loss_g1, error_bound_g0, error_bound_boosting, error_bound_g1,\
        error_bound_order_g0, error_bound_order_g1 = run_test_with_boosting(options_0, options_1, mode='ranking')

    # Run only with G2 training
    pretrained_g10_path, pretrained_g20_path = './saved_models/discogan_shoes2edges/g1_only', \
                                               './saved_models/discogan_shoes2edges/g2_one-sample_g1-fixed'
    data_a, data_b = dataset.get_edges2shoes()
    data_len = min(len(data_a), len(data_b))
    data_a, data_b = data_a[:data_len], data_b[:data_len]

    data_A_val, data_B_val = dataset.get_edges2shoes(test=True)
    J_loss_order_0, J_loss_val_0, ground_truth_loss_0 = initial_boosting_calc(pretrained_g10_path, pretrained_g20_path,
                                                                              data_a=data_a, data_b=data_b)
    data_a_reorder_0, data_b_reorder_0 = return_partial_ordered(data_a, data_b, J_loss_order_0, keep_portion=0.5)
    over_train = int(round(1 / (len(data_a_reorder_0) / len(data_a))))
    g1_1_path, g2_1_path = boosting_train_only_g2_train(pretrained_g10_path, pretrained_g20_path, data_a_reorder_0,
                                                        data_b_reorder_0,
                                                        data_A_val, data_B_val, phase=2, over_train=over_train,
                                                        boosting_phases=1,
                                                        b_weights=None)
    options_0, options_1 = define_options_for_inference(pretrained_g10_path, pretrained_g20_path,
                                                        pretrained_g10_path + '_boosting1_g1',
                                                        pretrained_g20_path + '_boosting1_g2', phase=2, over_train=2)

    gt_loss_g0, gt_loss_boosting, gt_loss_g1, error_bound_g0, error_bound_boosting, error_bound_g1,\
        error_bound_order_g0, error_bound_order_g1 = run_test_with_boosting(options_0, options_1, mode='ranking')

    # Run with no training

    pretrained_g10_path, pretrained_g20_path = './saved_models/discogan_shoes2edges/g1_only', \
                                               './saved_models/discogan_shoes2edges/g2_one-sample_g1-fixed'
    data_a, data_b = dataset.get_edges2shoes()
    data_len = min(len(data_a), len(data_b))
    data_a, data_b = data_a[:data_len], data_b[:data_len]

    data_A_val, data_B_val = dataset.get_edges2shoes(test=True)
    J_loss_order_0, J_loss_val_0, ground_truth_loss_0 = initial_boosting_calc(pretrained_g10_path, pretrained_g20_path,
                                                                              data_a=data_a, data_b=data_b)
    data_a_reorder_0, data_b_reorder_0 = return_partial_ordered(data_a, data_b, J_loss_order_0, keep_portion=0.5)
    over_train = int(round(1 / (len(data_a_reorder_0) / len(data_a))))
    g1_1_path, g2_1_path = boosting_no_train(pretrained_g10_path, pretrained_g20_path, data_a_reorder_0,
                                             data_b_reorder_0,
                                             data_A_val, data_B_val, phase=2, over_train=over_train, boosting_phases=1,
                                             b_weights=None)
    options_0, options_1 = define_options_for_inference(pretrained_g10_path, pretrained_g20_path,
                                                        pretrained_g10_path + '_boosting1_g1',
                                                        pretrained_g20_path + '_boosting1_g2', phase=2, over_train=2)

    gt_loss_g0, gt_loss_boosting, gt_loss_g1, error_bound_g0, error_bound_boosting, error_bound_g1,\
        error_bound_order_g0, error_bound_order_g1 = run_test_with_boosting(options_0, options_1, mode='ranking')