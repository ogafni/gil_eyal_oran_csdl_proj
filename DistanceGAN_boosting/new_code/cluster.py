from new_code import dataset
from new_code.options import Options
from new_code.clustering import Clustering

if __name__ == "__main__":
    options = Options()
    options.model_path='/home/gil/MEGA'
    options.task_name='task'
    options.epoch_size
    #options.cuda='false'
    options.number_of_samples=10000
    model = Clustering(options)
    data_a, data_b = dataset.get_edges2shoes(number_of_samples=options.number_of_samples)
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True, number_of_samples=options.batch_size)
    model.cluster(data_a, data_b, data_a_val, data_b_val)
