import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from progressbar import ProgressBar, Percentage, Bar, ETA

from .utils import to_no_grad_var
from .dataset import *


def calc_error_bound(samples, G1, G2, loss):
    G1_out = G1(samples)
    G2_out = G2(samples)
    return [loss(to_no_grad_var(G1_sample), to_no_grad_var(G2_sample)) for G1_sample, G2_sample in zip(G1_out, G2_out)]


def calc_mean_error_bound(samples, G1, G2, loss):
    bounds = calc_error_bound(samples, G1, G2, loss)
    return np.mean([bound.data[0] for bound in bounds])


def calc_gt_error(samples, labels, G1, loss):
    G1_out = G1(samples)
    return [loss(to_no_grad_var(G1_sample), to_no_grad_var(label)) for G1_sample, label in zip(G1_out, labels)]


def calc_mean_gt_error(samples, labels, G1, loss):
    errors = calc_gt_error(samples, labels, G1, loss)
    return np.mean([error.data[0] for error in errors])


def calc_error_bound_and_gt(samples, labels, G1, G2):
    """ 
    This function calculates the Error bound (correlation loss) per batch
    & the ground truth for G1
    """
    loss = nn.L1Loss()
    correlation_loss_AB_12 = calc_error_bound(samples, G1, G2, loss)
    ground_truth_loss_AB_G1 = calc_gt_error(samples, labels, G1, loss)
    return correlation_loss_AB_12, ground_truth_loss_AB_G1


def samples_order_by_loss(samples, labels, G1, G2, n_batch=64, print_freq=100):
    """
    This function sorts a dataset of samples by the error bound (correlation loss)
    It returns a vector of indices, representing the following:
    - loss order (max-->min)
    - loss vector (unordered)
    - ground truth loss vector
    """
    n_samples = len(samples)  # number of samples in the dataset
    print('Number of samples: ', n_samples)
    bounds = np.zeros(n_samples, dtype=float)  # initalize loss vec
    ground_truth_loss = np.zeros(n_samples, dtype=float)  # initalize ground truth loss vec
    n_iter = math.ceil(n_samples / n_batch)
    print('Number of iterations: ', n_iter)
    widgets = [Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=n_iter, widgets=widgets)
    pbar.start()

    for idx in np.arange(n_iter):
        pbar.update(idx)
        start = idx * n_batch
        end = min(start + n_batch, n_samples)
        samples_batch = samples[start: end, :, :, :].cuda()
        labels_batch = labels[start: end, :, :, :].cuda()
        bounds[start: end], ground_truth_loss[start: end] = calc_error_bound_and_gt(samples_batch, labels_batch, G1,G2)
    J_loss_order = bounds.argsort()[::-1]  # sort max-->min
    return J_loss_order, bounds, ground_truth_loss


def samples_order_by_loss_from_filenames(dataset_A, dataset_B, G1, G2, is_cuda=True, n_batch=64, print_freq=100):
    """
    enveloping function for the samples_order_by_loss, receiving list of filenames instead of arrays with images
    :param dataset_A: file names from dataset A
    :param dataset_B: file names from dataset B
    :param G1_0_A:
    :param G2_0_A:
    :param G1_0_B:
    :param G2_0_B:
    :param options: needed for creating a discogan class with correct task etc. so the internal samples_order_by_loss could work
    :param n_batch:
    :param print_freq:
    :return:
    """

    test_A = read_images(dataset_A, domain='A')
    test_B = read_images(dataset_B, domain='B')

    test_A = Variable(torch.FloatTensor(test_A))
    test_B = Variable(torch.FloatTensor(test_B))
    if is_cuda:
        test_A = test_A.cuda(0)
        test_B = test_B.cuda(0)
    return samples_order_by_loss(test_B, test_A, G1, G2, n_batch, print_freq)

def reorder_samples_by_loss(J_loss_order, samples):
    """
    This function received:
    - J_loss_order (the new loss order max-->min) 
    - S (Domain Dataset to be reordered)
    Example run-sequence:
        S_A_reordered = reorder_samples_by_loss(J_loss_order, S_A)
        S_A_reordered = Variable(S_A_reordered)
        S_B_reordered = reorder_samples_by_loss(J_loss_order, S_B)
        S_B_reordered = Variable(S_B_reordered)
        if model_discogan_with_risk.cuda:
            S_A_reordered = S_A_reordered.cuda(0)
            S_B_reordered = S_B_reordered.cuda(0)
    """
    return samples[J_loss_order.tolist()]
