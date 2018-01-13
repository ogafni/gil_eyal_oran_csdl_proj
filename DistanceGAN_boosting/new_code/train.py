
from new_code import dataset
from new_code.options import Options
from new_code.disco_gan_model import DiscoGAN

if __name__ == "__main__":
    options = Options()
    options.model_save_interval = 100
    options.model_path = '/home/gil/MEGA/temp/'
    options.result_path = '/home/gil/dl_wolf_git/gil_eyal_oran_csdl_proj/DistanceGAN_boosting/saved_runs/'
    model = DiscoGAN(options)
    data_a, data_b = dataset.get_edges2shoes()
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True, number_of_samples=options.batch_size)
    model.train(data_a, data_b, data_a_val, data_b_val)
