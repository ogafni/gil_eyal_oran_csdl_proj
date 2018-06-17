from new_code import dataset
from new_code.disco_boost import KeepBadWeighter, RoundsSC, DiscoBoost, ADAWeighter
from new_code.options import Options


def run_experiment(options, weighter, boosting_rounds):
    sc = RoundsSC(boosting_rounds)
    model = DiscoBoost(options, weighter, sc)
    data_a, data_b = dataset.get_data(options.dataset, test=False)
    data_a_val, data_b_val = dataset.get_data(options.dataset, test=True)
    model.train(data_a, data_b, data_a_val[:options.batch_size], data_b_val[:options.batch_size])


if __name__ == "__main__":
    options = Options(task_name='halving',
                      dataset='maps',
                      model_path='./models',
                      result_path='./results',
                      batch_size=8)
    split_weighter = KeepBadWeighter(0.5)
    run_experiment(options, split_weighter, 5)
    #ada_weighter = ADAWeighter()
    #run_experiment(options, ada_weighter, 3)