from new_code.disco_boost import KeepBadWeighter, RoundsSC, DiscoBoost
from new_code import dataset
from new_code.options import Options
import sys

# sys.path.insert(0, '/home/deepjunior/anaconda3/projects/gil_eyal_oran_csdl_proj/DistanceGAN_boosting')

if __name__ == "__main__": # These settings are used since G1 fixed training is not implemented yet
    # options = Options(task_name='boost_debug', dataset='edges2shoes', epoch_size=15, model_save_interval=25,
    #                   result_path='./results/', model_path='./models/', continue_training=False,
    #                   fixed_g1=True, pretrained_g1='./saved_models/reference/edges2shoes',
    #                   pretrained_g2='./saved_models/reference/edges2shoes', is_auto_detect_training_version=False)
    # options = Options(task_name='boost_kbw05_ba_load2_save3_gc1', dataset='edges2shoes', epoch_size=54,
    #                   result_path='./results/', model_path='./models/', continue_training=False,
    #                   fixed_g1=True, pretrained_g1='./saved_models/reference/edges2shoes', which_epoch_load=2,
    #                   pretrained_g2='./saved_models/reference/edges2shoes', is_auto_detect_training_version=False,
    #                   direction_btoa=True, gan_curriculum=1, version_save=2)
    options = Options(task_name='boost_kbw05_ba', dataset='edges2shoes', epoch_size=54,
                      result_path='./results/', model_path='./models/', continue_training=False,
                      fixed_g1=True, pretrained_g1='./saved_models/reference/edges2shoes', which_epoch_load=None,
                      pretrained_g2='./saved_models/reference/edges2shoes', is_auto_detect_training_version=False,
                      direction_btoa=True, gan_curriculum=1, version_save=2)
    weighter = KeepBadWeighter(0.5)
    sc = RoundsSC(3)
    model = DiscoBoost(options, weighter, sc)

    # data_a, data_b = dataset.get_edges2shoes(test=False, number_of_samples=1000)
    data_a, data_b = dataset.get_edges2shoes(test=False)
    data_len = min(len(data_a), len(data_b))
    data_a, data_b = data_a[:data_len], data_b[:data_len]
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True)
    model.train(data_a, data_b, data_a_val[:options.batch_size], data_b_val[:options.batch_size])
    model.infer(data_a_val, data_b_val, method=0)  # see method options within function