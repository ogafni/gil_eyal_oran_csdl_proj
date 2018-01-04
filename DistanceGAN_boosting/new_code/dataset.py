import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
shoe_path = os.path.join(dataset_path, 'edges2shoes')


def read_images(filenames, domain=None, image_size=64, split=256):
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

        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(image)

    images = np.stack(images)
    return images


def get_edges2shoes(test=False, number_of_samples=None):
    if test:
        item_path = os.path.join(shoe_path, 'val')
    else:
        item_path = os.path.join(shoe_path, 'train')

    image_paths = [os.path.join(item_path, x) for x in os.listdir(item_path)]

    if number_of_samples is not None:
        image_paths = image_paths[:number_of_samples]

    if test:
        return [image_paths, image_paths]
    else:
        n_images = len(image_paths)
        mid = round(n_images / 2)
        return [image_paths[:mid], image_paths[mid:]]


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


def get_data_loaders(data_a, data_b, batch_size, b_weights=None):
    a_dataset = DomainAdaptationDataset(data_a)
    b_dataset = DomainAdaptationDataset(data_b)

    a_dataloader = DataLoader(a_dataset, batch_size=batch_size, shuffle=True)
    # Weights should only apply to B
    sampler = WeightedRandomSampler(b_weights, len(b_weights)) if b_weights else None
    b_dataloader = DataLoader(b_dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler == None)
    return a_dataloader, b_dataloader
