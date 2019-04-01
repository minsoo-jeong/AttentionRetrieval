import torch.autograd.variable as V
import torch

import shutil
import os

from utils.AverageMeter import AverageMeter


def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters():
            p.requires_grad = False


def train(model, loader, optim, criterion, ep, iter, verbose=0):
    losses = AverageMeter()
    pos = AverageMeter()
    neg = AverageMeter()

    model.train()
    model.apply(set_bn_to_eval)
    print('>> Train\tEpoch {}\tlr {}'.format(ep, optim.param_groups[0]['lr']))

    for idx, batch in enumerate(loader):
        p, q, n = batch
        optim.zero_grad()

        out1, out2, out3 = model(V(q[0]).cuda(), V(p[0]).cuda(), V(n[0]).cuda())
        loss, dp, dn = criterion(anchor=out1[0], positive=out2[0], negative=out3[0], size_average=True)
        loss.backward()
        optim.step()

        losses.update(loss.item(), n=p[0].size(0))
        pos.update(dp.item(), n=p[0].size(0))
        neg.update(dn.item(), n=p[0].size(0))

        if verbose and iter % verbose == 0:
            print('Iter {} Epoch {} [{}/{}] Loss {:.4f}({:.4f}) Pos {:.4f}({:.4f}) Neg {:.4f}({:.4f})'
                  .format(iter, ep, idx * loader.batch_size + p[0].size(0), len(loader.dataset),
                          loss.item(), losses.avg, dp.item(), pos.avg, dn.item(), neg.avg))

        iter += 1

    print('>> [ Train\tEpoch {} - Loss {:.4f} Pos {:.4f} Neg {:.4f} ]'.format(ep, losses.avg, pos.avg, neg.avg))
    return losses.avg, pos.avg, neg.avg, iter


def valid(model, loader, criterion, ep, iter, verbose=0):
    losses = AverageMeter()
    pos = AverageMeter()
    neg = AverageMeter()

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            p, q, n = batch

            out1, out2, out3 = model(V(q[0]).cuda(), V(p[0]).cuda(), V(n[0]).cuda())
            loss, dp, dn = criterion(anchor=out1[0], positive=out2[0], negative=out3[0], size_average=True)

            losses.update(loss.item(), n=p[0].size(0))
            pos.update(dp.item(), n=p[0].size(0))
            neg.update(dn.item(), n=p[0].size(0))

            if verbose and iter % verbose == 0:
                print('Iter {} Epoch {} [{}/{}] Loss {:.4f}({:.4f}) Pos {:.4f}({:.4f}) Neg {:.4f}({:.4f})'
                      .format(iter, ep, idx * loader.batch_size + p[0].size(0), len(loader.dataset),
                              loss.item(), losses.avg, dp.item(), pos.avg, dn.item(), neg.avg))

            iter += 1

    print('>> [ Valid\tEpoch {} - Loss {:.4f} Pos {:.4f} Neg {:.4f} ]'.format(ep, losses.avg, pos.avg, neg.avg))
    return losses.avg, pos.avg, neg.avg, iter


def save_ckpt(state, is_best, path):
    if not os.path.exists(path):
        os.makedirs(path)
    file = os.path.join(path, 'model_epoch_{}.pth.tar'.format(state['epoch'] - 1))
    torch.save(state, file)
    print('save ckpt .... {}'.format(file))
    if is_best:
        file_best = os.path.join(path, 'model_best.pth.tar'.format(state['epoch'] - 1))
        shutil.copyfile(file, file_best)
        print('save best ckpt .... {}'.format(file_best))
