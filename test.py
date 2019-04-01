import torchvision.transforms as trn
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import torch

import numpy as np
import pickle
import os

from Network import dataset


class GroundTruth:
    def __init__(self, img_root, gt_file):
        self.gt_file = gt_file
        self.name = os.path.splitext(self.gt_file)[0].split('/gnd_')[-1]
        with open(gt_file, 'rb') as f:
            gnd = pickle.load(f)

        self.imlist = gnd['imlist']
        self.qimlist = gnd['qimlist']
        self.gnd = gnd['gnd']
        self.bbx = [i['bbx'] for i in gnd['gnd']]
        self.real_imlist = list(map(lambda x: os.path.join(img_root, '{}.jpg'.format(x)), self.imlist))
        self.real_qimlist = list(map(lambda x: os.path.join(img_root, '{}.jpg'.format(x)), self.qimlist))


def extractor_vector(model, loader):
    print('extract vector...', end='')
    vecs = []
    with torch.no_grad():
        for n, i in enumerate(loader):
            out, _ = model(V(i[0]).cuda())
            vecs.append(out.cpu().data)
            if (n + 1) % 10 == 0 or (n + 1) == len(loader.dataset):
                print('\r>>>> {}/{} done...'.format(n * loader.batch_size + out.size(0), len(loader.dataset)), end='')
        print('\r', end='')
        vecs = torch.cat(vecs, 0).numpy()
    return vecs


def compute_ap(ranks, nres):
    # number of images ranked by the system
    nimgranks = len(ranks)
    # accumulate trapezoids in PR-plot
    ap = 0
    recall_step = 1. / nres
    for j in np.arange(nimgranks):
        rank = ranks[j]
        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.
    return ap


def compute_map(ranks, gnd, kappas=[]):
    mAP = 0.  # mean average precision
    num_querys = len(gnd)
    aps = np.zeros(num_querys)  # average precision per query
    pr = np.zeros(len(kappas))  # mean precision at kappas
    prs = np.zeros((num_querys, len(kappas)))  # precision at each kappas
    nempty = 0

    for i in range(num_querys):
        q_gnd_ok = np.array(gnd[i]['ok'])

        # 0 gnd['ok']
        if q_gnd_ok.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            q_gnd_junk = np.array(gnd[i]['junk'])
        except:
            # 0 gnd['junk']
            q_gnd_junk = np.empty(0)

        # get positive and negative in ranks[i]
        pos = np.arange(ranks.shape[1])
        pos = pos[np.in1d(ranks[i], q_gnd_ok)]
        junk = np.arange(ranks.shape[1])
        junk = junk[np.in1d(ranks[i], q_gnd_junk)]

        k = 0
        ij = 0
        if len(junk):
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        ap = compute_ap(pos, len(q_gnd_ok))
        mAP = mAP + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    mAP = mAP / (num_querys - nempty)
    pr = pr / (num_querys - nempty)

    return mAP, aps, pr, prs


def compute_map_and_print(ranks, gnd, kappas=[1, 5, 10]):
    if gnd.name == 'oxford5k' or gnd.name == 'paris6k':
        map, aps, _, _ = compute_map(ranks, gnd.gnd)
        print('>> {}: mAP {:.2f}'.format(gnd.name, np.around(map * 100, decimals=2)))

    else:
        num_querys = len(gnd.qimlist)
        gnd_t = []
        for i in range(num_querys):
            g = {}
            g['ok'] = np.concatenate([gnd.gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd.gnd[i]['junk'], gnd.gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks=ranks, gnd=gnd_t, kappas=kappas)
        # print(mapE, apsE, mprE, prsE)
        gnd_t = []
        for i in range(num_querys):
            g = {}
            g['ok'] = np.concatenate([gnd.gnd[i]['easy'], gnd.gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd.gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)
        # print(mapM, apsM, mprM, prsM)
        gnd_t = []
        for i in range(num_querys):
            g = {}
            g['ok'] = np.concatenate([gnd.gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd.gnd[i]['junk'], gnd.gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)
        # print(mapH, apsH, mprH, prsH)

        print('>> {} : mAP E: {}, M: {}, H: {}'.format(gnd.name, np.around(mapE * 100, decimals=2),
                                                       np.around(mapM * 100, decimals=2),
                                                       np.around(mapH * 100, decimals=2)))
        print('>> {} : mP@k{} E: {}, M: {}, H: {}'.format(gnd.name, kappas, np.around(mprE * 100, decimals=2),
                                                          np.around(mprM * 100, decimals=2),
                                                          np.around(mprH * 100, decimals=2)))
        map = mapM
    return map


def test(model, gnd, image_size, transform):
    model.cuda()
    model.eval()

    im = dataset.ListDataSet(gnd.real_imlist, image_size=image_size, bbx=None, transform=transform)
    query = dataset.ListDataSet(gnd.real_qimlist, image_size=image_size, bbx=gnd.bbx, transform=transform)

    im_loader = DataLoader(im, batch_size=1, shuffle=False, num_workers=8)
    q_loader = DataLoader(query, batch_size=1, shuffle=False, num_workers=4)

    im_vecs = extractor_vector(model, im_loader)
    q_vecs = extractor_vector(model, q_loader)

    score = np.dot(q_vecs, im_vecs.T)
    ranks = np.argsort(-score, axis=1)

    map = compute_map_and_print(ranks, gnd)
    return map


if __name__ == '__main__':
    '''
    paris = GroundTruth(os.path.join('/data', 'paris6k', 'jpg'),
                        os.path.join('/data', 'paris6k', 'gnd_paris6k.pkl'))
    query = dataset.ListDataSet(paris.real_qimlist, bbx=paris.bbx, transform=None)
    print(query[0][0].size(), query[0][1], paris.bbx[0])

    import PIL.Image
    import torchvision.transforms as trn

    tr = trn.ToPILImage()
    i = tr((query[0][0]))

    im = PIL.Image.open('/data/paris6k/jpg/paris_defense_000605.jpg')
    im.convert('RGB')
    print(im)
    im.thumbnail((1024, 1024), PIL.Image.ANTIALIAS)
    print(im)
    '''
    import numpy as np
    from models import nets, pooling
    import torchvision.transforms as trn

    qvecs = np.load('/qvecs-rmac.npy').T
    vecs = np.load('/vecs-rmac.npy').T

    print(qvecs.shape)
    print(vecs.shape)

    score = np.dot(qvecs, vecs.T)
    ranks = np.argsort(-score, axis=1)

    paris = GroundTruth(os.path.join('/data', 'paris6k', 'jpg'),
                        os.path.join('/data', 'paris6k', 'gnd_paris6k.pkl'))
    compute_map_and_print(ranks, paris)

    embed = nets.Basic(pooling.RMAC())
    transform_test = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    embed.cuda()
    embed.eval()

    im = dataset.ListDataSet(paris.real_imlist, image_size=224, bbx=None, transform=transform_test)
    query = dataset.ListDataSet(paris.real_qimlist, image_size=None, bbx=paris.bbx, transform=transform_test)
    print(im[0][0].size())
    print(im[1][0].size())
    print(im[2][0].size())
    print(query[0][0].size(), query[0][1])
    print(query[1][0].size(), query[1][1])
    print(query[2][0].size(), query[2][1])
    q_loader = DataLoader(query, batch_size=1, shuffle=False, num_workers=4)

    # q_vecs = extractor_vector(embed, q_loader)
    # print(q_vecs)
    # print(qvecs)
