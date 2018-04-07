#from new_code.disco_boost import KeepBadWeighter, RoundsSC, DiscoBoost
from new_code import dataset
from new_code.disco_gan_model import DiscoGAN
from new_code.options import Options
import sys


if __name__ == "__main__": # These settings are used since G1 fixed training is not implemented yet
    options = Options(task_name='cluster_A1_B5', dataset='edges2shoes', model_path='/home/gil/MEGA/saved_models',
                      result_path='/home/gil/dl_wolf_git/gil_eyal_oran_csdl_proj/DistanceGAN_boosting/results',epoch_size=700)
    disco_gan = DiscoGAN(options)
    data_a, data_b = dataset.get_edges2shoes(test=False, cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',cluster_idx_A=1,
                                             cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
                                             cluster_idx_B=5)
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True, cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',cluster_idx_A=1,
                                             cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
                                             cluster_idx_B=5)
    #disco_gan.train(data_a[:1000],data_b[:1000],data_a_val[:500],data_b_val[:500])
    disco_gan.train(data_a, data_b, data_a_val, data_b_val)
    # weighter = KeepBadWeighter(0.7)
    # sc = RoundsSC(3)
    # model = DiscoBoost(options, weighter, sc)
    # data_a, data_b = dataset.get_edges2shoes(test=False, number_of_samples=1000)
    # data_a_val, data_b_val = dataset.get_edges2shoes(test=True)
    # model.train(data_a, data_b, data_a_val[:options.batch_size], data_b_val[:options.batch_size])
    # model.infer(data_a_val, data_b_val)