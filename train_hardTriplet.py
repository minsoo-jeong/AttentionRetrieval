import torch.autograd.variable as V
import torch.nn.functional as F
import torch

import numpy as np

import shutil
import os

from utils.AverageMeter import AverageMeter
from utils.whiten import *

import logging


def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters():
            p.requires_grad = False


def train(model, loader, optim, criterion, ep, iter, verbose=0):
    log = logging.getLogger('my')
    losses = AverageMeter()
    pos = AverageMeter()
    neg = AverageMeter()
    nTriplet = AverageMeter()

    model.module.base.apply(set_bn_to_eval)
    model.train()
    log.info('>> Train\tEpoch {}\tlr {}'.format(ep, optim.param_groups[0]['lr']))
    gamma=[0.25, 0.3536, 0.5000, 0.7072, 1.0]
    for idx, batch in enumerate(loader):
        optim.zero_grad()
        g = np.random.choice(5, 1)[0]
        bat = F.interpolate(batch[0], scale_factor=gamma[g], mode='bilinear')
        feature, attention = model(V(bat).cuda())
        #feature, attention = model(V(batch[0]).cuda())
        loss, dp, dn, n_triplet = criterion(embeddings=feature, target=batch[1])
        loss.backward()
        optim.step()

        losses.update(loss.item(), n=1)
        pos.update(dp.item(), n=1)
        neg.update(dn.item(), n=1)
        nTriplet.update(n_triplet, n=1)

        if verbose and iter % verbose == 0:
            print(
                '\tIter {}\tEpoch {}\t[{}/{}]\tLoss {:.4f}({:.4f})\tPos {:.4f}({:.4f})\tNeg {:.4f}({:.4f})\tTriplet {:.2f}({:.2f})'
                    .format(iter, ep, idx * loader.batch_sampler.batch_size + batch[1].size(0), len(loader.dataset),
                            loss.item(), losses.avg, dp.item(), pos.avg, dn.item(), neg.avg, n_triplet, nTriplet.avg))

        iter += 1

    log.info(
        '>> [ Train\tIter {}\tEpoch {}\tLoss {:.4f}\tPos {:.4f}\tNeg {:.4f}\tTriplet {:.2f} ]'.format(iter,ep, losses.avg, pos.avg,
                                                                                             neg.avg, nTriplet.avg))
    return losses.avg, pos.avg, neg.avg, nTriplet.avg, iter


def valid(model, loader, criterion, ep, iter, verbose=0):
    log = logging.getLogger('my')
    losses = AverageMeter()
    pos = AverageMeter()
    neg = AverageMeter()
    nTriplet = AverageMeter()
    feats=[]
    model.eval()
    gamma=[0.25, 0.3536, 0.5000, 0.7072, 1.0]
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            g = np.random.choice(5, 1)[0]
            bat = F.interpolate(batch[0], scale_factor=gamma[g], mode='bilinear')
            feature, attention = model(V(bat).cuda())
            #feature, attention = model(V(batch[0]).cuda())
            loss, dp, dn, n_triplet = criterion(embeddings=feature, target=batch[1])
            feats.append(feature.data.cpu().numpy())
            losses.update(loss.item(), n=1)
            pos.update(dp.item(), n=1)
            neg.update(dn.item(), n=1)
            nTriplet.update(n_triplet, n=1)

            if verbose and iter % verbose == 0:
                print(
                    '\tIter {}\tEpoch {}\t[{}/{}]\tLoss {:.4f}({:.4f})\tPos {:.4f}({:.4f})\tNeg {:.4f}({:.4f})\tTriplet {:.2f}({:.2f})'
                    .format(iter, ep, idx * loader.batch_sampler.batch_size + batch[1].size(0), len(loader.dataset),
                            loss.item(), losses.avg, dp.item(), pos.avg, dn.item(), neg.avg, n_triplet,
                            nTriplet.avg))

            iter += 1
    feats=np.concatenate(feats)
    whiten=pcawhitenlearn(feats)
    log.info(
        '>> [ Valid\titer {}\tEpoch {}\tLoss {:.4f}\tPos {:.4f}\tNeg {:.4f}\tTriplet {:.2f} ]'.format(iter,ep, losses.avg, pos.avg,
                                                                                             neg.avg, nTriplet.avg))
    return losses.avg, pos.avg, neg.avg, nTriplet.avg, iter,whiten


def save_ckpt(state, is_best=False, path='./ckpts'):
    log = logging.getLogger('my')
    if not os.path.exists(path):
        os.makedirs(path)
    file = os.path.join(path, 'model_epoch_{}.pth.tar'.format(state['epoch'] - 1))
    torch.save(state, file)
    log.info('>> save ckpt .... {}'.format(file))
    if is_best:
        file_best = os.path.join(path, 'model_best.pth.tar')
        shutil.copyfile(file, file_best)
        log.info('save best ckpt .... {}'.format(file_best))
