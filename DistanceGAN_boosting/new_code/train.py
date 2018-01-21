from new_code.disco_boost import KeepBadWeighter, RoundsSC, DiscoBoost
from new_code import dataset
from new_code.options import Options

if __name__ == "__main__":
    options = Options()
    weighter = KeepBadWeighter(0.7)
    sc = RoundsSC(4)
    model = DiscoBoost(options, weighter, sc)
    data_a, data_b = dataset.get_edges2shoes()
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True, number_of_samples=options.batch_size)
    model.train(data_a, data_b, data_a_val, data_b_val)
