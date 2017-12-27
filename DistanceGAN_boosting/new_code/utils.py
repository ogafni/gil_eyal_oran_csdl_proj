import torch
from torch.autograd import Variable
import numpy as np

from .dataset import read_images as read_dataset_images


def to_no_grad_var(var):
    data = as_np(var)
    is_cuda = var.is_cuda
    var = Variable(torch.FloatTensor(data), requires_grad=False)
    if is_cuda:
        var = var.cuda(0)
    return var


def as_np(data):
    return data.cpu().data.numpy()


def get_fm_loss(real_feats, fake_feats, feat_criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = feat_criterion(l2, Variable(torch.ones(l2.size())).cuda())
        losses += loss

    return losses


def get_gan_loss(dis_real, dis_fake, gan_criterion, cuda):
    labels_dis_real = Variable(torch.ones([dis_real.size()[0], 1]))
    labels_dis_fake = Variable(torch.zeros([dis_fake.size()[0], 1]))
    labels_gen = Variable(torch.ones([dis_fake.size()[0], 1]))

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = gan_criterion(dis_real, labels_dis_real) * 0.5 + gan_criterion(dis_fake, labels_dis_fake) * 0.5
    gen_loss = gan_criterion(dis_fake, labels_gen)

    return dis_loss, gen_loss


def read_images(A, B, image_size, cuda, aligned=False):
    if aligned:
        A = np.concatenate([A, B])
        B = A
    A = read_dataset_images(A, 'A', image_size)
    B = read_dataset_images(B, 'B', image_size)
    A = Variable(torch.FloatTensor(A))
    B = Variable(torch.FloatTensor(B))
    if cuda:
        A = A.cuda()
        B = B.cuda()
    return A, B


def get_model_l2(model):
    params = [param for _, param in model.named_parameters()]
    return sum([torch.norm(param, p=2) for param in params]) / len(params)


def get_batch_data(A, B, batch, batch_size):
    data_size = min(len(A), len(B))
    i_start = batch_size * batch
    i_end = min(i_start + batch_size, data_size)
    return A[i_start:i_end], B[i_start:i_end]
