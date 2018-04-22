#from new_code.disco_boost import KeepBadWeighter, RoundsSC, DiscoBoost
from new_code import dataset
from new_code.disco_gan_model import DiscoGAN
from new_code.options import Options
import sys


if __name__ == "__main__": # These settings are used since G1 fixed training is not implemented yet
    # options = Options(task_name='cluster_A2_B4', dataset='edges2shoes', model_path='/home/gil/MEGA/saved_models',
    #                   result_path='/home/gil/dl_wolf_git/gil_eyal_oran_csdl_proj/DistanceGAN_boosting/results',epoch_size=800)
    # disco_gan = DiscoGAN(options)
    # data_a, data_b = dataset.get_edges2shoes(test=False, cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',cluster_idx_A=2,
    #                                          cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
    #                                          cluster_idx_B=4)
    # data_a_val, data_b_val = dataset.get_edges2shoes(test=True, cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',cluster_idx_A=2,
    #                                          cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
    #                                          cluster_idx_B=4)
    #
    # disco_gan.train(data_a, data_b, data_a_val, data_b_val)
    #
    # options = Options(task_name='cluster_A3_B1', dataset='edges2shoes', model_path='/home/gil/MEGA/saved_models',
    #                   result_path='/home/gil/dl_wolf_git/gil_eyal_oran_csdl_proj/DistanceGAN_boosting/results',
    #                   epoch_size=800)
    # disco_gan = DiscoGAN(options)
    # data_a, data_b = dataset.get_edges2shoes(test=False,
    #                                          cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',
    #                                          cluster_idx_A=3,
    #                                          cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
    #                                          cluster_idx_B=1)
    # data_a_val, data_b_val = dataset.get_edges2shoes(test=True,
    #                                                  cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',
    #                                                  cluster_idx_A=3,
    #                                                  cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
    #                                                  cluster_idx_B=1)
    #
    # disco_gan.train(data_a, data_b, data_a_val, data_b_val)

    # options = Options(task_name='cluster_A4_B3', dataset='edges2shoes', model_path='/home/gil/MEGA/saved_models',
    #                   result_path='/home/gil/dl_wolf_git/gil_eyal_oran_csdl_proj/DistanceGAN_boosting/results',
    #                   epoch_size=800)
    # disco_gan = DiscoGAN(options)
    # data_a, data_b = dataset.get_edges2shoes(test=False,
    #                                          cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',
    #                                          cluster_idx_A=4,
    #                                          cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
    #                                          cluster_idx_B=3)
    # data_a_val, data_b_val = dataset.get_edges2shoes(test=True,
    #                                                  cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',
    #                                                  cluster_idx_A=4,
    #                                                  cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
    #                                                  cluster_idx_B=3)
    #
    # disco_gan.train(data_a, data_b, data_a_val, data_b_val)

    options = Options(task_name='cluster_A5_B0', dataset='edges2shoes', model_path='/home/gil/MEGA/saved_models',
                      result_path='/home/gil/dl_wolf_git/gil_eyal_oran_csdl_proj/DistanceGAN_boosting/results',
                      epoch_size=800)
    disco_gan = DiscoGAN(options)
    data_a, data_b = dataset.get_edges2shoes(test=False,
                                             cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',
                                             cluster_idx_A=5,
                                             cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
                                             cluster_idx_B=0)
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True,
                                                     cluster_file_A='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_A.pkl',
                                                     cluster_idx_A=5,
                                                     cluster_file_B='/home/gil/MEGA/saved_models/referece/edges2shoes/kmenas_B.pkl',
                                                     cluster_idx_B=0)

    disco_gan.train(data_a, data_b, data_a_val, data_b_val)
