import os
import cv2
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
shoe_path = os.path.join(dataset_path, 'edges2shoes')
handbag_path = os.path.join(dataset_path, 'edges2handbags')
maps_path = os.path.join(dataset_path, 'maps')
city_path = os.path.join(dataset_path, 'cityscapes')


def read_images(filenames, domain=None, image_size=64, split=256, is_gray = False):
    images = []

    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue

        if domain == 'A':
            kernel = np.ones((3, 3), np.uint8)
            image = image[:, :split, :]
            image = 255. - image
            image = cv2.dilate(image, kernel, iterations=1)
            image = 255. - image
        elif domain == 'B':
            image = image[:, split:, :]
            if is_gray:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(image)

    images = np.stack(images)
    return images


def get_maps(test=False, number_of_samples=None):
    return _get_data(maps_path, test, number_of_samples)


def get_city_scapes(test=False, number_of_samples=None):
    return _get_data(city_path, test, number_of_samples, test_dir='test')


def get_edges2shoes(test=False, number_of_samples=None, cluster_file=None, cluster_idx=0, is_cluster_A=False):
    return _get_data(shoe_path, test, number_of_samples, cluster_file=cluster_file, cluster_idx=cluster_idx, is_cluster_A=is_cluster_A)


def get_edges2handbags(test=False, number_of_samples=None, ):
    return _get_data(handbag_path, test, number_of_samples)


def _get_data(item_path, test=False, number_of_samples=None, test_dir='val', cluster_file=None, cluster_idx=0, is_cluster_A=False):
    item_path = os.path.join(item_path, test_dir if test else 'train')

    image_paths = [os.path.join(item_path, x) for x in os.listdir(item_path)]

    if number_of_samples is not None:
        image_paths = image_paths[:number_of_samples]

    if test:
        if cluster_file:
            image_paths = filter_by_cluster(image_paths, cluster_file, cluster_idx)
        return [image_paths, image_paths]
    else:
        n_images = len(image_paths)
        mid = round(n_images / 2)
        data_a = image_paths[:mid]
        data_b = image_paths[mid:]
        if cluster_file:
            if is_cluster_A:
                data_a = filter_by_cluster(data_a, cluster_file, cluster_idx)
            else:
                data_b = filter_by_cluster(data_b, cluster_file, cluster_idx)
        return [data_a, data_b]




class DomainAdaptationDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        self.dataset_len = len(files)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        sample = self.files[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_data_loaders(data_a, data_b, batch_size, b_weights=None, is_shuffle = True):
    a_dataset = DomainAdaptationDataset(data_a)
    b_dataset = DomainAdaptationDataset(data_b)
    a_dataloader = DataLoader(a_dataset, batch_size=batch_size, shuffle=is_shuffle)
    # Weights should only apply to B
    sampler = None if b_weights is None else WeightedRandomSampler(b_weights, len(b_weights))
    b_dataloader = DataLoader(b_dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler == None and is_shuffle))
    return a_dataloader, b_dataloader

def filter_by_cluster(dataset, cluster_file, cluster_idx):
    with open(cluster_file, 'rb') as f:
        codebook, labels, all_names = pickle.load(f)
    cur_dir=os.path.dirname(dataset[0])
    cluster_dataset=[]
    labels = np.asarray(labels)
    rel_idx = np.where(labels == cluster_idx)[0]
    rel_names = [os.path.join(cur_dir,all_names[x]) for x in rel_idx]
    for file in rel_names:
        if file in dataset:
            cluster_dataset.append(file)

    return cluster_dataset

