import numpy as np
from .disco_gan_model import DiscoGAN
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

class Clustering():
    def __init__(self, options):
        self.gan = DiscoGAN(options)
        self.args = options
        self.model_path = self.args.get_model_path()


    def _calc_dis_activation(self, discriminator_A, discriminator_B, A, B):
        A_dis_real, A_feats_real = discriminator_A(A)
        B_dis_real, B_feats_real = discriminator_B(B)
        return A_feats_real, B_feats_real

    def cluster(self, data_A, data_B, is_cluster_A=False):

            n_batches = math.ceil(len(data_A) / self.args.batch_size)
            print('%d batches per epoch' % n_batches)
            all_feats = np.empty(shape=(0,512*16))
            #all_names = []
            a_dataloader, b_dataloader = get_data_loaders(data_A, data_B, self.args.batch_size,None,False)
            is_gray = True
            if is_cluster_A:
                save_postfix='_A'
            else:
                save_postfix='_B'
            if not os.path.exists(os.path.join(self.model_path, 'dis_feats'+save_postfix+'.npy')):
                for i, (A_paths, B_paths) in enumerate(zip(a_dataloader, b_dataloader)):

                    # read batch data
                    A, B = read_images(A_paths, B_paths, self.args.image_size, self.args.cuda, self.args.dataset, False, is_gray)

                    A_feats_real, B_feats_real = self._calc_dis_activation(self.gan.discriminator_A, self.gan.discriminator_B, A, B)
                    #temp = np.mean(B_feats_real[2].cpu().data.numpy(),axis=(2,3))
                    if is_cluster_A:
                        temp = A_feats_real[2].cpu().data.numpy()
                    else:
                        temp = B_feats_real[2].cpu().data.numpy()
                    temp = temp.reshape((temp.shape[0],temp.shape[1]*temp.shape[2]*temp.shape[3]))
                    all_feats=np.concatenate((all_feats,temp))
                    #all_names += B_paths
                    print(i)
                np.save(os.path.join(self.model_path, 'dis_feats'+save_postfix+'.npy'),all_feats)

            else:
                all_feats = np.load(os.path.join(self.model_path, 'dis_feats'+save_postfix+'.npy'))

            #pca = PCA(n_components=512)
            #pca.fit(all_feats)
            #white_feats = whiten(pca.transform(all_feats))
            if is_cluster_A:
                all_names = [os.path.basename(x) for x in a_dataloader.dataset.files]
            else:
                all_names = [os.path.basename(x) for x in b_dataloader.dataset.files]
            white_feats = whiten(all_feats)
            k = self.args.k_clusters
            if not os.path.exists(os.path.join(self.model_path, 'kmenas'+save_postfix+'.pkl')):
                codebook, labels = kmeans2(white_feats, k)
                with open(os.path.join(self.model_path, 'kmenas'+save_postfix+'.pkl'), 'wb') as f:
                    pickle.dump([codebook, labels, all_names], f)
            else:
                with open(os.path.join(self.model_path, 'kmenas'+save_postfix+'.pkl'), 'rb') as f:
                    codebook, labels, all_names = pickle.load(f)

            os.makedirs(os.path.join(self.model_path, 'clusters'+save_postfix),exist_ok=True)
            for idx in range(k):
                os.makedirs(os.path.join(self.model_path,'clusters'+save_postfix,str(idx)),exist_ok=True)

            save_count = np.zeros(k)
            if is_cluster_A:
                data = data_A
            else:
                data = data_B

            for idx, file in enumerate(data):
                save_count[labels[idx]] += 1
                if save_count[labels[idx]]>100:
                    continue
                copyfile(file,os.path.join(self.model_path,'clusters'+save_postfix,str(labels[idx]),os.path.basename(file)))





