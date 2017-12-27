from . import dataset
from .options import Options
from .disco_gan_model import DiscoGAN

if __name__ == "__main__":
    options = Options()
    model = DiscoGAN(options)
    data_a, data_b = dataset.get_edges2shoes()
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True, number_of_samples=options.batch_size)
    model.train(data_a, data_b, data_a_val, data_b_val)
