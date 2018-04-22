#from new_code.disco_boost import KeepBadWeighter, RoundsSC, DiscoBoost
from new_code import dataset
from new_code.disco_gan_model import DiscoGAN
from new_code.options import Options
from new_code.error_bound_calc_functions import *
import sys



def load_model(filename):
    model = torch.load(filename)
    model = model.cuda(0)
    model.train(False)
    return model

def load_models(options):
    generator_A_1 = load_model(os.path.join(options.pretrained_g1, 'gen_A-' + str(options.which_epoch_load)))
    generator_B_1 = load_model(os.path.join(options.pretrained_g1, 'gen_B-' + str(options.which_epoch_load)))

    return generator_A_1, generator_B_1

def samples_gt_error(samples, labels, G1, n_batch=64):
    """
    This function sorts a dataset of samples by the error bound (correlation loss)
    It returns a vector of indices, representing the following:
    - loss order (max-->min)
    - loss vector (unordered)
    - ground truth loss vector
    """
    n_samples = len(samples)  # number of samples in the dataset
    print('Calculating bounds of {0} samples'.format(n_samples))
    ground_truth_loss = np.zeros(n_samples, dtype=float)  # initalize ground truth loss vec
    n_iter = math.ceil(n_samples / n_batch)
    widgets = [Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=n_iter, widgets=widgets)
    pbar.start()
    loss = nn.L1Loss()

    for idx in np.arange(n_iter):
        pbar.update(idx)
        start = idx * n_batch
        end = min(start + n_batch, n_samples)
        samples_batch = samples[start: end, :, :, :].cuda()
        labels_batch = labels[start: end, :, :, :].cuda()
        ground_truth_loss[start: end] = calc_gt_error(samples_batch, labels_batch, G1, loss)
    return ground_truth_loss

def samples_gt_error_from_filenames(dataset_A, dataset_B, G1, is_cuda=True, n_batch=64, direcion_btoa=True):
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
    if direcion_btoa:
        return samples_gt_error(test_B, test_A, G1, n_batch)
    else:
        return samples_gt_error(test_A, test_B, G1, n_batch)


def calc_gt_loss_of_cluster(cluster_idx_A, cluster_idx_B, cluster_file_A, cluster_file_B, pretrained_g1_path):

    options = Options(pretrained_g1=pretrained_g1_path)

    if cluster_idx_A == -1 or cluster_idx_B == -1:
        data_a_val, data_b_val = dataset.get_edges2shoes(test=True)
    else:
        data_a_val, data_b_val = dataset.get_edges2shoes(test=True,
                                                     cluster_file_A=cluster_file_A,
                                                     cluster_idx_A=cluster_idx_A,
                                                     cluster_file_B=cluster_file_B,
                                                     cluster_idx_B=cluster_idx_B)

    G1_0_A, G1_0_B = load_models(options)

    groud_truth_loss = samples_gt_error_from_filenames(data_a_val, data_b_val, G1_0_A, options.cuda, options.batch_size)
    return groud_truth_loss


if __name__ == "__main__":

    pretrained_g1_path = '/home/gil/MEGA/saved_models/referece/edges2shoes/G1'
    print(np.average(calc_gt_loss_of_cluster(-1, -1, '', '', pretrained_g1_path)))

    cluster_file_A = '/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl'
    cluster_file_B = '/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl'

    groud_truth_loss=[]

    cluster_idx_A = [0,1,2,3,4,5]
    cluster_idx_B = [2,5,4,1,3,0]

    for idx in range(6):

        pretrained_g1_path='/home/gil/MEGA/saved_models/cluster_A'+str(cluster_idx_A[idx])+'_B'+str(cluster_idx_B[idx])+\
                           '/edges2shoes/G1'

        groud_truth_loss += calc_gt_loss_of_cluster(cluster_idx_A[idx], cluster_idx_B[idx], cluster_file_A, cluster_file_B, pretrained_g1_path).tolist()

    print(np.average(np.asarray(groud_truth_loss)))

