import torchvision.transforms as trn
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import torch

import numpy as np
import pickle
import os

from utils.whiten import *
from Network import dataset
from models.nets import *
import logging


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

    def get_visulize_list(self, n_cls):
        qim_idx = np.random.choice(len(self.real_qimlist), n_cls, replace=False)
        l = []
        for q in qim_idx:
            l.append(self.real_qimlist[q])
            if self.name == 'rparis6k' or self.name == 'roxford5k':
                l.append(self.real_imlist[self.gnd[q]['easy'][0]])
                l.append(self.real_imlist[self.gnd[q]['hard'][0]])
                l.append(self.real_imlist[self.gnd[q]['hard'][1]])

        return l


def extractor_vector(model, loader):
    print('extract vector...', end='')
    model.eval()
    vecs = []
    with torch.no_grad():
        for n, i in enumerate(loader):
            out, _ = model(V(i[0]).cuda())
            vecs.append(out.cpu().data)
            if (n + 1) % 50 == 0 or (n + 1) == len(loader.dataset):
                print('\r>>>> {}/{} done...'.format(n * loader.batch_size + out.size(0), len(loader.dataset)), end='')
        print('\r', end='')
        vecs = torch.cat(vecs, 0).numpy()
    return vecs

def test_basic(model, gnd, image_size, transform,whiten):
    log = logging.getLogger('my')
    model.cuda()
    model.eval()
    im = dataset.ListDataSet(gnd.real_imlist, image_size=image_size, bbx=None, transform=transform)
    query = dataset.ListDataSet(gnd.real_qimlist, image_size=image_size, bbx=None, transform=transform)
    query_bbx = dataset.ListDataSet(gnd.real_qimlist, image_size=image_size, bbx=gnd.bbx, transform=transform)

    im_loader = DataLoader(im, batch_size=1, shuffle=False, num_workers=4)
    q_loader = DataLoader(query, batch_size=1, shuffle=False, num_workers=4)
    qbbx_loader = DataLoader(query_bbx, batch_size=1, shuffle=False, num_workers=4)

    q_vecs = extractor_vector(model, q_loader)
    qbbx_vecs = extractor_vector(model, qbbx_loader)
    im_vecs = extractor_vector(model, im_loader)

    score = np.dot(q_vecs, im_vecs.T)
    ranks = np.argsort(-score, axis=1)
    map, msg = compute_map_and_print(ranks, gnd)
    log.info('{} - q(no Bbox) : {}'.format(gnd.name, msg))
    print('{} - q(no Bbox) : {}'.format(gnd.name, msg))
    score_bbx_attn = np.dot(qbbx_vecs, im_vecs.T)
    ranks_bbx_attn = np.argsort(-score_bbx_attn, axis=1)
    map3, msg = compute_map_and_print(ranks_bbx_attn, gnd)
    log.info('{} - q(Bbox) : {}'.format(gnd.name, msg))
    print('{} - q(Bbox) : {}'.format(gnd.name, msg))
    map = max(map, map3)
    map_whiten = 0
    if whiten is not None:
        log.info('>> whiten with landmark clean-valid'.format(gnd.name, msg))
        im_vecs = whitenapply(im_vecs, whiten[0], whiten[1])
        q_vecs = whitenapply(q_vecs, whiten[0], whiten[1])
        qbbx_vecs = whitenapply(qbbx_vecs, whiten[0], whiten[1])

        score = np.dot(q_vecs, im_vecs.T)
        ranks = np.argsort(-score, axis=1)
        map_whiten, msg = compute_map_and_print(ranks, gnd)
        log.info('{} - q(no Bbox) : {}'.format(gnd.name, msg))
        print('{} - q(no Bbox) : {}'.format(gnd.name, msg))
        score_bbx_attn = np.dot(qbbx_vecs, im_vecs.T)
        ranks_bbx_attn = np.argsort(-score_bbx_attn, axis=1)
        map_whiten3, msg = compute_map_and_print(ranks_bbx_attn, gnd)
        log.info('{} - q(Bbox) : {}'.format(gnd.name, msg))
        print('{} - q(Bbox) : {}'.format(gnd.name, msg))
        map_whiten = max(map_whiten, map_whiten3)
    map = max(map, map_whiten)
    return map

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
    log = logging.getLogger('my')
    if gnd.name == 'oxford5k' or gnd.name == 'paris6k':
        map, aps, _, _ = compute_map(ranks, gnd.gnd)
        # print('>> {}: mAP {:.2f}'.format(gnd.name, np.around(map * 100, decimals=2)))
        msg = 'mAP {:.2f}'.format(np.around(map * 100, decimals=2))

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

        msg = 'mAP E: {}, M: {}, H: {} / mP@k{} E: {}, M: {}, H: {}' \
            .format(np.around(mapE * 100, decimals=2), np.around(mapM * 100, decimals=2),
                    np.around(mapH * 100, decimals=2), kappas, np.around(mprE * 100, decimals=2),
                    np.around(mprM * 100, decimals=2), np.around(mprH * 100, decimals=2))

        # log.info('>> {} : mAP E: {}, M: {}, H: {}'.format(gnd.name, np.around(mapE * 100, decimals=2), np.around(mapM * 100, decimals=2),  np.around(mapH * 100, decimals=2)))
        # log.info('>> {} : mP@k{} E: {}, M: {}, H: {}'.format(gnd.name, kappas, np.around(mprE * 100, decimals=2), np.around(mprM * 100, decimals=2),np.around(mprH * 100, decimals=2)))
        map = mapM
    return map, msg


def test(model, basic, gnd, image_size, transform,whiten):
    log = logging.getLogger('my')
    model.cuda()
    model.eval()
    basic.base.load_state_dict(model.module.base.state_dict())
    basic.pool.load_state_dict(model.module.pool.state_dict())
    basic.cuda()
    basic.eval()

    im = dataset.ListDataSet(gnd.real_imlist, image_size=image_size, bbx=None, transform=transform)
    query = dataset.ListDataSet(gnd.real_qimlist, image_size=image_size, bbx=None, transform=transform)
    query_bbx = dataset.ListDataSet(gnd.real_qimlist, image_size=image_size, bbx=gnd.bbx, transform=transform)

    im_loader = DataLoader(im, batch_size=1, shuffle=False, num_workers=4)
    q_loader = DataLoader(query, batch_size=1, shuffle=False, num_workers=4)
    qbbx_loader = DataLoader(query_bbx, batch_size=1, shuffle=False, num_workers=4)

    q_vecs = extractor_vector(model, q_loader)
    qbbx_vecs = extractor_vector(model, qbbx_loader)
    qbbx_basic_vecs = extractor_vector(basic, qbbx_loader)
    im_vecs = extractor_vector(model, im_loader)

    score = np.dot(q_vecs, im_vecs.T)
    ranks = np.argsort(-score, axis=1)
    map, msg = compute_map_and_print(ranks, gnd)
    log.info('{} - q(Attn) : {}'.format(gnd.name, msg))
    print('{} - q(Attn) : {}'.format(gnd.name, msg))
    score_bbx_basic = np.dot(qbbx_basic_vecs, im_vecs.T)
    ranks_bbx_basic = np.argsort(-score_bbx_basic, axis=1)

    map2, msg = compute_map_and_print(ranks_bbx_basic, gnd)
    log.info('{} - q(Bbox) : {}'.format(gnd.name, msg))
    print('{} - q(Bbox) : {}'.format(gnd.name, msg))
    score_bbx_attn = np.dot(qbbx_vecs, im_vecs.T)
    ranks_bbx_attn = np.argsort(-score_bbx_attn, axis=1)
    map3, msg = compute_map_and_print(ranks_bbx_attn, gnd)
    log.info('{} - q(Bbox+Attn) : {}'.format(gnd.name, msg))
    print('{} - q(Bbox+Attn) : {}'.format(gnd.name, msg))
    map = max(max(map, map2), map3)
    map_whiten = 0
    if whiten is not None:
        log.info('>> whiten with landmark clean-valid'.format(gnd.name, msg))
        im_vecs = whitenapply(im_vecs, whiten[0], whiten[1])
        q_vecs = whitenapply(q_vecs, whiten[0], whiten[1])
        qbbx_vecs = whitenapply(qbbx_vecs, whiten[0], whiten[1])
        qbbx_basic_vecs = whitenapply(qbbx_basic_vecs, whiten[0], whiten[1])

        score = np.dot(q_vecs, im_vecs.T)
        ranks = np.argsort(-score, axis=1)
        map_whiten, msg = compute_map_and_print(ranks, gnd)
        log.info('{} - q(Attn) : {}'.format(gnd.name, msg))

        score_bbx_basic = np.dot(qbbx_basic_vecs, im_vecs.T)
        ranks_bbx_basic = np.argsort(-score_bbx_basic, axis=1)

        map_whiten2, msg = compute_map_and_print(ranks_bbx_basic, gnd)
        log.info('{} - q(Bbox) : {}'.format(gnd.name, msg))

        score_bbx_attn = np.dot(qbbx_vecs, im_vecs.T)
        ranks_bbx_attn = np.argsort(-score_bbx_attn, axis=1)
        map_whiten3, msg = compute_map_and_print(ranks_bbx_attn, gnd)
        log.info('{} - q(Bbox+Attn) : {}'.format(gnd.name, msg))
        map_whiten = max(max(map_whiten, map_whiten2), map_whiten3)
    map = max(map, map_whiten)
    return map


def get_imvec(model, im_loader, path):
    f = np.sort(os.listdir(path))
    f = [os.path.join(path, i) for i in f]
    vecs = []
    model.eval()
    with torch.no_grad():
        for n, i in enumerate(f):
            out, _ = model(V(torch.Tensor(torch.load(i))).cuda())
            vecs.append(out.cpu().data)
            if (n + 1) % 50 == 0 or (n + 1) == len(im_loader.dataset):
                print('\r>>>> {}/{} done...'.format(n * im_loader.batch_size + out.size(0), len(im_loader.dataset)),
                      end='')
        print('\r', end='')
        vecs = torch.cat(vecs, 0).numpy()

    return vecs


def test_prefetch(model, basic, preattn, prebasic, gnd, image_size, transform, whiten,path):
    log = logging.getLogger('my')
    model.cuda()
    model.eval()
    basic.base.load_state_dict(model.module.base.state_dict())
    basic.pool.load_state_dict(model.module.pool.state_dict())
    basic.cuda()
    basic.eval()

    preattn.attention.load_state_dict(model.module.attention.state_dict())
    preattn.pool.load_state_dict(model.module.pool.state_dict())
    preattn.cuda()
    preattn = nn.DataParallel(preattn)
    preattn.eval()

    prebasic.pool.load_state_dict(model.module.pool.state_dict())
    prebasic.cuda()
    prebasic = nn.DataParallel(prebasic)
    prebasic.eval()

    im = dataset.ListDataSet(gnd.real_imlist, image_size=image_size, bbx=None, transform=transform)
    query = dataset.ListDataSet(gnd.real_qimlist, image_size=image_size, bbx=None, transform=transform)
    query_bbx = dataset.ListDataSet(gnd.real_qimlist, image_size=image_size, bbx=gnd.bbx, transform=transform)

    im_loader = DataLoader(im, batch_size=1, shuffle=False, num_workers=4)
    q_loader = DataLoader(query, batch_size=1, shuffle=False, num_workers=4)
    qbbx_loader = DataLoader(query_bbx, batch_size=1, shuffle=False, num_workers=4)

    im_vecs = get_imvec(preattn, im_loader, path)
    q_vecs = extractor_vector(model, q_loader)
    qbbx_vecs = extractor_vector(model, qbbx_loader)
    qbbx_basic_vecs = extractor_vector(basic, qbbx_loader)

    score = np.dot(q_vecs, im_vecs.T)
    ranks = np.argsort(-score, axis=1)
    map, msg = compute_map_and_print(ranks, gnd)
    log.info('{} - q(Attn) : {}'.format(gnd.name, msg))

    score_bbx_basic = np.dot(qbbx_basic_vecs, im_vecs.T)
    ranks_bbx_basic = np.argsort(-score_bbx_basic, axis=1)

    map2, msg = compute_map_and_print(ranks_bbx_basic, gnd)
    log.info('{} - q(Bbox) : {}'.format(gnd.name, msg))

    score_bbx_attn = np.dot(qbbx_vecs, im_vecs.T)
    ranks_bbx_attn = np.argsort(-score_bbx_attn, axis=1)
    map3, msg = compute_map_and_print(ranks_bbx_attn, gnd)
    log.info('{} - q(Bbox+Attn) : {}'.format(gnd.name, msg))
    map = max(max(map, map2), map3)
    map_whiten=0
    if whiten is not None:
        log.info('>> whiten with landmark clean-valid'.format(gnd.name, msg))
        im_vecs = whitenapply(im_vecs,whiten[0],whiten[1])
        q_vecs = whitenapply(q_vecs,whiten[0],whiten[1])
        qbbx_vecs = whitenapply(qbbx_vecs,whiten[0],whiten[1])
        qbbx_basic_vecs = whitenapply(qbbx_basic_vecs,whiten[0],whiten[1])

        score = np.dot(q_vecs, im_vecs.T)
        ranks = np.argsort(-score, axis=1)
        map_whiten, msg = compute_map_and_print(ranks, gnd)
        log.info('{} - q(Attn) : {}'.format(gnd.name, msg))

        score_bbx_basic = np.dot(qbbx_basic_vecs, im_vecs.T)
        ranks_bbx_basic = np.argsort(-score_bbx_basic, axis=1)

        map_whiten2, msg = compute_map_and_print(ranks_bbx_basic, gnd)
        log.info('{} - q(Bbox) : {}'.format(gnd.name, msg))

        score_bbx_attn = np.dot(qbbx_vecs, im_vecs.T)
        ranks_bbx_attn = np.argsort(-score_bbx_attn, axis=1)
        map_whiten3, msg = compute_map_and_print(ranks_bbx_attn, gnd)
        log.info('{} - q(Bbox+Attn) : {}'.format(gnd.name, msg))
        map_whiten = max(max(map_whiten, map_whiten2), map_whiten3)
    map=max(map,map_whiten)
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
    from models import nets, pooling,attention
    import torchvision.transforms as trn

    model=nets.Attn(attention.AttentionCS(pooling.RMAC()),pooling.RMAC())
    basic = nets.Basic(pooling.RMAC())

    ckpts=torch.load('/data/ckpts/RMAC/3D-cs/Cpool-rmac-Sigmoid/freeze9/lr-1e-05-margin-0.2-rndm720/model_epoch_4.pth.tar')
    model.load_state_dict(ckpts['model_state_dict'])
    basic.base.load_state_dict(model.base.state_dict())
    basic.pool.load_state_dict(model.pool.state_dict())
    model.cuda()
    model=nn.DataParallel(model)
    #model.base.load_state_dict(model_zoo.load_url('http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth'))
    #model.base.load_state_dict(nn.Sequential(*list(resnet50(pretrained=True).children())[:-2]).state_dict())
    oxford = GroundTruth(os.path.join('/data', 'oxford5k', 'jpg'),
                          os.path.join('/data', 'oxford5k', 'gnd_oxford5k.pkl'))
    paris = GroundTruth(os.path.join('/data', 'paris6k', 'jpg'),
                         os.path.join('/data', 'paris6k', 'gnd_paris6k.pkl'))
    roxford = GroundTruth(os.path.join('/data', 'roxford5k', 'jpg'),
                          os.path.join('/data', 'roxford5k', 'gnd_roxford5k.pkl'))
    rparis = GroundTruth(os.path.join('/data', 'rparis6k', 'jpg'),
                         os.path.join('/data', 'rparis6k', 'gnd_rparis6k.pkl'))

    test_trn = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # map = test_basic(model, paris, 1024, test_trn, None)
    oxd=dataset.ListDataSet(oxford.real_qimlist, image_size=1024, bbx=None, transform=test_trn)
    ox = DataLoader(oxd, batch_size=1, shuffle=False)
    oxv = extractor_vector(model, ox)
    np.save('out/oxford_q.npy',oxv)

    pad = dataset.ListDataSet(paris.real_qimlist, image_size=1024, bbx=None, transform=test_trn)
    pa = DataLoader(pad, batch_size=1, shuffle=False)
    pav = extractor_vector(model, pa)
    np.save('out/paris_q.npy', pav)

    roxd = dataset.ListDataSet(roxford.real_qimlist, image_size=1024, bbx=None, transform=test_trn)
    rox = DataLoader(roxd, batch_size=1, shuffle=False)
    roxv = extractor_vector(model, rox)
    np.save('out/roxford_q.npy', roxv)

    rpad = dataset.ListDataSet(rparis.real_qimlist, image_size=1024, bbx=None, transform=test_trn)
    rpa = DataLoader(rpad, batch_size=1, shuffle=False)
    rpav = extractor_vector(model, rpa)
    np.save('out/rparis_q.npy', rpav)
    # map=test(model,basic,roxford,1024,test_t

    '''
    qvecs = np.load('/qvecs-rmac.npy').T
    vecs = np.load('/vecs-rmac.npy').T

    print(qvecs.shape)
    print(vecs.shape)

    score = np.dot(qvecs, vecs.T)
    ranks = np.argsort(-score, axis=1)


    rparis = GroundTruth(os.path.join('/data', 'rparis6k', 'jpg'),
                         os.path.join('/data', 'rparis6k', 'gnd_rparis6k.pkl'))
    print(rparis.get_visulize_list(3))
    print(rparis.qimlist[30], rparis.gnd[30])
    print(rparis.qimlist[0], rparis.imlist[85], rparis.imlist[114])
    print(rparis.qimlist[5], rparis.imlist[41], rparis.imlist[3547])
    print(rparis.qimlist[30], rparis.imlist[50], rparis.imlist[513])
    embed = nets.Basic(pooling.RMAC())
    transform_test = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    embed.cuda()
    embed.eval()

    im = dataset.ListDataSet(paris.real_imlist, image_size=224, bbx=None, transform=transform_test)
    query = dataset.ListDataSet(paris.real_qimlist, image_size=None, bbx=paris.bbx, transform=transform_test)
    print(im[:5])
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
    '''