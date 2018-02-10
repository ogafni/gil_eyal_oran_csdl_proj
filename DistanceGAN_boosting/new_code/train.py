from new_code.disco_boost import KeepBadWeighter, RoundsSC, DiscoBoost
from new_code import dataset
from new_code.options import Options
import sys


if __name__ == "__main__": # These settings are used since G1 fixed training is not implemented yet
    options = Options(task_name='boost_debug', dataset='edges2shoes', epoch_size=5, model_save_interval=25,
                      result_path='./results/', model_path='./models/', continue_training=False, indiv_gan_rate=1,
                      fixed_g1=False, start_from_pretrained_g1=False, start_from_pretrained_g2=False,
                      is_auto_detect_training_version=True, direction_btoa=True)
    weighter = KeepBadWeighter(0.7)
    sc = RoundsSC(3)
    model = DiscoBoost(options, weighter, sc)
    data_a, data_b = dataset.get_edges2shoes(test=False, number_of_samples=1000)
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True)
    model.train(data_a, data_b, data_a_val[:options.batch_size], data_b_val[:options.batch_size])
    model.infer(data_a_val, data_b_val)