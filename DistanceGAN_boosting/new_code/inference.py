from new_code.options import Options
from new_code.dataset import *
from new_code.error_bound_calc_functions import *
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
    generator_A_1 = load_model(os.path.join(options.pretrained_g1_path_A, 'model_gen_A_G1-' + str(options.which_epoch_load)))
    generator_B_1 = load_model(os.path.join(options.pretrained_g1_path_B, 'model_gen_B_G1-' + str(options.which_epoch_load)))
    generator_A_2 = load_model(os.path.join(options.pretrained_g2_path_A, 'model_gen_A_G2-' + str(options.which_epoch_load)))
    generator_B_2 = load_model(os.path.join(options.pretrained_g2_path_B, 'model_gen_B_G2-' + str(options.which_epoch_load)))

    return generator_A_1, generator_B_1, generator_A_2, generator_B_2


def run_test_with_boosting(options1, options2, mode='thresh', thresh=None):
    #loading the test data, in an ugly but quick way

    test_style_A, test_style_B = get_edges2shoes(test = True)


    #loading the models for 2 iterations of boosting
    G1_0_A, G1_0_B, G2_0_A, G2_0_B = load_models(options1)
    G1_1_A, G1_1_B, G2_1_A, G2_1_B = load_models(options2)

    #running G1_0 G2_0 on test data, and ordering by loss
    J_loss_order, J_loss_val, groud_truth_loss = samples_order_by_loss_from_filenames(test_style_A, test_style_B, G1_0_A, G2_0_A,options1.cuda,
                                                                                      options1.batch_size)

    # run the second generators on the test set
    J_loss_order2, J_loss_val2, groud_truth_loss2 = samples_order_by_loss_from_filenames(test_style_A, test_style_B,
                                                                                         G1_1_A, G2_1_A,
                                                                                         options2.cuda,
                                                                                         options2.batch_size)

    if mode == 'thresh':
        #if we got a Threshold from the training time, use it, if not - take the median of the loss on the test set as the threshold
        if thresh is None:
            thresh = np.median(J_loss_val)

        # idx's of images from the first round, that the error is lower than threshold
        first_round_idx = np.where(J_loss_val <= thresh)
        #idx's that the error is bigger than threshold, on these images we will run the boosting
        second_round_idx = np.where(J_loss_val > thresh)
    elif mode == 'ranking':
        first_round_idx = np.asarray([i for i in range(200) if np.where(J_loss_order == i)[0] > np.where(J_loss_order2 == i)[0]])
        second_round_idx = np.asarray(
            [i for i in range(200) if np.where(J_loss_order == i)[0] <= np.where(J_loss_order2 == i)[0]])




    #calculate the average ground truth error if we didn't have boosting
    original_ground_truth_loss = np.average(groud_truth_loss)

    #calculate the average error with the boosting. taking the ground truth loss of first iteration only on idx's of first round,
    #and adding the ground truth lost from the second iteration on the second iteration idx's
    boosting_ground_truth_loss = np.concatenate((groud_truth_loss[first_round_idx],groud_truth_loss2[second_round_idx]))
    mean_boosting_ground_truth_loss = np.average(boosting_ground_truth_loss)
    
    J_loss_boost =  np.concatenate((J_loss_val[first_round_idx],J_loss_val2[second_round_idx]))
    
    #hopefully the boosting loss will be smaller than the original loss
    print("Original Loss: {} Boosting Loss: {}".format(original_ground_truth_loss, mean_boosting_ground_truth_loss))
    return groud_truth_loss, boosting_ground_truth_loss, groud_truth_loss2, J_loss_val, J_loss_boost, J_loss_val2, J_loss_order, J_loss_order2


if __name__ == "__main__":
    gt_loss_g0, gt_loss_boosting, gt_loss_g1, error_bound_g0, error_bound_boosting, error_bound_g1, error_bound_order_g0, error_bound_order_g1 = run_test_with_boosting(options1, options2, mode = 'ranking')
    



