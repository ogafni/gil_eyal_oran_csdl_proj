def calc_error_bound_and_gt(samples, labels, G1_AB, G2_AB, G1_BA, G2_BA):
    """ 
    This function calculates the Error bound (correlation loss) per batch
    & the ground truth for G1
    """
    l1_loss_over_batch = nn.L1Loss()
    correlation_loss_AB_12 = np.zeros((len(samples)))
    ground_truth_loss_AB_G1 = np.zeros((len(labels)))
    AB_1 = G1_AB(samples)
    AB_2 = G2_AB(samples)
    n_samples = len(samples)

    for sample_idx in np.arange(n_samples):
        ### IN THE ORIGINAL SCRIPT, THE CORRELATION & GROUND TRUTH LOST WERE MULTIPLIED BY -1 
        correlation_loss_AB_12[sample_idx] =\
        model_discogan_with_risk.correlation_criterion(\
                                                       model_discogan_with_risk.to_no_grad_var(AB_1[sample_idx]),\
                                                       model_discogan_with_risk.to_no_grad_var(AB_2[sample_idx]))
        ground_truth_loss_AB_G1[sample_idx] =\
        model_discogan_with_risk.correlation_criterion(\
                                                       model_discogan_with_risk.to_no_grad_var(AB_1[sample_idx]),\
                                                       model_discogan_with_risk.to_no_grad_var(labels[sample_idx]))
    return correlation_loss_AB_12, ground_truth_loss_AB_G1

def samples_order_by_loss(S_A, S_B, G1_0_A, G2_0_A, G1_0_B, G2_0_B, n_batch=64, print_freq=100):
    """
    This function sorts a dataset of samples by the error bound (correlation loss)
    It returns a vector of indices, representing the following:
    - loss order (max-->min)
    - loss vector (unordered)
    - ground truth loss vector
    Example run-line: 
        J_loss_order, J_loss_val, groud_truth_loss = samples_order_by_loss(S_A, S_B, G1_0_A, G2_0_A, G1_0_B, G2_0_B, n_batch=64, print_freq=100)
    """
    n_samples = len(S_A) # number of samples in the dataset
    print('Number of samples: ',n_samples)
    J_loss_val = np.zeros(n_samples, dtype=float) # initalize loss vec
    groud_truth_loss = np.zeros(n_samples, dtype=float) # initalize ground truth loss vec
    n_iter = n_samples // n_batch
    print('Number of iterations: ',n_iter)
    for idx in np.arange(n_iter): 
        J_loss_val[idx*n_batch: (idx+1)*n_batch], groud_truth_loss[idx*n_batch: (idx+1)*n_batch] =\
        calc_error_bound_and_gt(S_A[idx*n_batch: (idx+1)*n_batch, :, :, :],\
                                S_B[idx*n_batch: (idx+1)*n_batch, :, :, :],\
                                G1_0_A, G2_0_A, G1_0_B, G2_0_B) # calculate error bound (loss) per batch
        if idx % print_freq == 0: # printing frequency
            print('Done calculating ', idx*n_batch, ' samples')
    print ('Error bound & ground truth calculated for all samples')
    J_loss_order = J_loss_val.argsort()[::-1][:len(J_loss_val)] # sort max-->min
    return J_loss_order, J_loss_val, groud_truth_loss 

### Vectorized implementation (Issues with Pytorch Variable?)
#
# def reorder_samples_by_loss(J_loss_order, S):
#     """
#     This function received:
#     - J_loss_order (the new loss order max-->min) 
#     - S (Domain Dataset to be reordered)
#     """
#     loss_order = torch.from_numpy(J_loss_order.copy()).cuda()
#     S_new = torch.Tensor(S.size()).cuda()
#     S_new.index_copy_(0, loss_order, S.data)
#     S_new = Variable(S_new)
#     return S_new

### Trivial implementation (loop)
def reorder_samples_by_loss(J_loss_order, S):
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
    S_new = torch.Tensor(S.size()).cuda()
    for idx in np.arange(S.size()[0]):
        S_new[idx, :, :, :] = S.data[J_loss_order[idx], :, :, :]
    return S_new