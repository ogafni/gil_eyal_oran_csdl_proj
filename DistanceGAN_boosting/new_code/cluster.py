from new_code import dataset
from new_code.options import Options
from new_code.clustering import Clustering

if __name__ == "__main__":
    options = Options()
    options.model_path='/home/gil/MEGA/saved_models'
    options.task_name='referece'
    options.k_clusters = 6
    #options.not_all_samples = True
    #options.number_of_samples=1000
    model = Clustering(options)
    data_a, data_b = dataset.get_edges2shoes(number_of_samples=options.number_of_samples if options.not_all_samples else None)
    data_a_val, data_b_val = dataset.get_edges2shoes(test=True)
    model.cluster(data_a+data_a_val, data_b+data_b_val)
