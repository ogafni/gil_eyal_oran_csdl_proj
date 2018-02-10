import numpy as np
from .disco_gan_model import DiscoGAN, get_generators, get_discriminators
import os
import copy
import numpy as np
import math
from .utils import *
from .model import Discriminator, Generator
from .dataset import get_data_loaders
from .disco_gan_with_risk import DiscoGANRisk
from .error_bound_calc_functions import samples_order_by_loss_from_filenames
from scipy.cluster.vq import kmeans2, whiten
from PIL import Image
from shutil import copyfile
import pickle
from sklearn.decomposition import PCA

class Clustering(DiscoGAN):
    def __init__(self, options):
        super().__init__(options)

    def _calc_dis_activation(self, discriminator_A, discriminator_B, A, B):
        A_dis_real, A_feats_real = discriminator_A(A)
        B_dis_real, B_feats_real = discriminator_B(B)
        return A_feats_real, B_feats_real

    def cluster(self, data_A, data_B, data_A_val, data_B_val):

            n_batches = math.ceil(len(data_A) / self.args.batch_size)
            print('%d batches per epoch' % n_batches)
            all_feats = np.empty(shape=(0,512*16))
            a_dataloader, b_dataloader = get_data_loaders(data_A, data_B, self.args.batch_size,None,False)
            is_gray = True
            if not os.path.exists(os.path.join(self.model_path, 'dis_feats.npy')):
                for i, (A_paths, B_paths) in enumerate(zip(a_dataloader, b_dataloader)):

                    # read batch data
                    A, B = read_images(A_paths, B_paths, self.args.image_size, self.cuda, self.args.dataset, is_gray)

                    A_feats_real, B_feats_real = self._calc_dis_activation(self.discriminator_A, self.discriminator_B, A, B)
                    #temp = np.mean(B_feats_real[2].cpu().data.numpy(),axis=(2,3))
                    temp = B_feats_real[2].cpu().data.numpy()
                    temp = temp.reshape((temp.shape[0],temp.shape[1]*temp.shape[2]*temp.shape[3]))
                    all_feats=np.concatenate((all_feats,temp))
                    print(i)
                np.save(os.path.join(self.model_path, 'dis_feats.npy'),all_feats)

            else:
                all_feats = np.load(os.path.join(self.model_path, 'dis_feats.npy'))

            #pca = PCA(n_components=512)
            #pca.fit(all_feats)
            #white_feats = whiten(pca.transform(all_feats))

            white_feats = whiten(all_feats)
            k = 10
            if not os.path.exists(os.path.join(self.model_path, 'kmenas.pkl')):
                codebook, labels = kmeans2(white_feats, k)
                with open(os.path.join(self.model_path, 'kmenas.pkl'), 'wb') as f:
                    pickle.dump([codebook, labels], f)
            else:
                with open(os.path.join(self.model_path, 'kmenas.pkl'), 'rb') as f:
                    codebook, labels = pickle.load(f)

            os.makedirs(os.path.join(self.model_path, 'clusters'),exist_ok=True)
            for idx in range(k):
                os.makedirs(os.path.join(self.model_path,'clusters',str(idx)),exist_ok=True)

            save_count = np.zeros(k)
            for idx, file in enumerate(data_B):
                save_count[labels[idx]] += 1
                if save_count[labels[idx]]>100:
                    continue
                copyfile(file,os.path.join(self.model_path,'clusters',str(labels[idx]),os.path.basename(file)))





