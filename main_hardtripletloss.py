import torchvision.transforms as trn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim import SGD
import torch.nn as nn

from datetime import datetime
import warnings
import os

from models import nets, pooling, attention
from Network.losses import OnlineTripletLoss, HardestNegativeTripletSelector
from Network.losses import *
from Network.batch_sampler import MaxBatchSampler, ClassBalanceSampler
from utils.TlsSMTPHandler import TlsSMTPHandler

from test import GroundTruth, test, test_prefetch, test_basic
from train_hardTriplet import *

from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

warnings.filterwarnings('ignore')
from PIL import Image

import logging
import time

from Network.dataset import ListDataSet
from utils.visualize import showAtmp, showAtmp_3D_ver2, showAtmp_3D
import argparse


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except:
        with open(path, 'r') as f:
            img = Image.open(f)
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class myFolder(DatasetFolder):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        super(myFolder, self).__init__(root, loader, extensions, transform, target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target
        except Exception as e:
            idx = index - 1 if index != 0 else index + 1


def init_logger(log_path):
    os.environ['TZ'] = 'Asia/Seoul'
    time.tzset()
    base_dir = os.path.abspath(os.path.dirname(log_path))
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    log = logging.getLogger('my')
    log.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s : %(message)s')
    formatter = logging.Formatter('[ %(asctime)s ] %(message)s')
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    smpt_handler = TlsSMTPHandler(("smtp.naver.com", 587), 'jms8167@naver.com', ['jms8167@gmail.com'], 'Error found!',
                                  ('jms8167', 's011435a!'))
    smpt_handler.setLevel(logging.ERROR)
    smpt_handler.setFormatter(formatter)

    log.addHandler(stream_handler)
    log.addHandler(file_handler)
    log.addHandler(smpt_handler)

    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.01)
    parser.add_argument("--margin", help="Triplet margin", type=float, default=0.2)
    parser.add_argument("--verbose", help="verbose per iterations", type=int, default=0)
    args = parser.parse_args()

    # torch.cuda.set_device(1)
    Epoch = 15
    start_epoch = 1
    it_train = it_valid = 0
    method = 'hardest-negative-triplet'
    lr = 0.001
    loss_margin = 0.2
    batch_size = 64
    n_visualize_cls = 3
    if args.lr:
        lr = args.lr
    if args.margin:
        loss_margin = args.margin
    verbose = args.verbose

    meta = 'lr-{}-margin-{}-rndm720'.format(lr, loss_margin)

    ckpt_path = os.path.join('/data', 'ckpts', 'RMAC', '3D-cs', 'Cpool-rmac-Sigmoid', 'freeze10', meta)
    visualize_path = os.path.join('./visualize', 'RMAC', '3D-cs', 'Cpool-rmac-Sigmoid', 'freeze10', meta)
    log_path = os.path.join('./logs', 'RMAC', '3D-cs', 'Cpool-rmac-Sigmoid', 'freeze10', '{}.txt'.format(meta))
    #ckpt_path = os.path.join('/data', 'ckpts', 'RMAC', 'base4', meta)
    #visualize_path = os.path.join('./visualize', 'RMAC', 'base4', meta)
    #log_path = os.path.join('./logs', 'RMAC', 'base4', '{}.txt'.format(meta))

    log = init_logger(log_path)
    log.info("\n==== Meta ====\nlog : {}\nckpt : {}\nloss : {}\nlr : {}\nmargin : {}\nbatch size : {}\n=============="
             .format(log_path, ckpt_path, method, lr, loss_margin, batch_size))

    # dataset
    log.info('>> Dataset ... ')
    # train / valid DataSet

    transform_valid = transform_train = trn.Compose([
        trn.Resize(900),
        trn.CenterCrop(900),
        trn.RandomCrop(720),
        #trn.Resize(350),
        #trn.RandomCrop(280),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainSet = myFolder(os.path.join('/landmark', 'Landmark-clean', 'train'), default_loader, ['jpg'],
                        transform=transform_train)

    trainLoader = DataLoader(trainSet, batch_sampler=ClassBalanceSampler(trainSet.targets, batch_size, 4),
                             num_workers=8)

    validSet = myFolder(os.path.join('/landmark', 'Landmark-clean', 'valid'), default_loader, ['jpg'],
                        transform=transform_train)

    validLoader = DataLoader(validSet, batch_sampler=ClassBalanceSampler(validSet.targets, batch_size, 4),
                             num_workers=8)

    # Test DataSet
    test_trn = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rparis = GroundTruth(os.path.join('/data', 'rparis6k', 'jpg'),
                         os.path.join('/data', 'rparis6k', 'gnd_rparis6k.pkl'))

    roxford = GroundTruth(os.path.join('/data', 'roxford5k', 'jpg'),
                          os.path.join('/data', 'roxford5k', 'gnd_roxford5k.pkl'))

    paris = GroundTruth(os.path.join('/paris6k', 'jpg'),
                        os.path.join('/paris6k', 'gnd_paris6k.pkl'))

    oxford = GroundTruth(os.path.join('/oxford5k', 'jpg'),
                         os.path.join('/oxford5k', 'gnd_oxford5k.pkl'))

    # make model
    log.info('>> Create Model ...  base')

    pool = pooling.RMAC()
    trained = nets.Basic(pool)
    trained.load_state_dict(torch.load('/data/ckpts/RMAC/base4/lr-0.001-margin-0.2-rndm256/model_epoch_16.pth.tar')['model_state_dict'])
    log.info('>> load base model : /data/ckpts/RMAC/base4/lr-0.001-margin-0.2-rndm256/model_epoch_16.pth.tar')

    model = nets.Attn(attention.AttentionCS2(pool=pooling.RMAC()), pooling.RMAC())
    model.base.load_state_dict(trained.base.state_dict())
    model.pool.load_state_dict(trained.pool.state_dict())

    basic = nets.Basic(model.pool)
    basic.base.load_state_dict(model.base.state_dict())
    basic.pool.load_state_dict(model.pool.state_dict())

    prebasic = nets.Prefetch_Basic(model.pool)
    preattn = nets.Prefetch_Attn(model.attention, model.pool)


    '''
    # unfreeze backbone
    log.info('>> unfreeze backbone')
    for n, p in model.base.named_parameters():
        p.requires_grad = True
    '''
    # freeze backbone
    log.info('>> freeze backbone')
    for n, p in model.base.named_parameters():
        p.requires_grad = False
    # initialize attention
    log.info('>> initialize attention module')
    for n, p in model.attention.named_modules():
        if isinstance(p, torch.nn.Conv2d):
            nn.init.xavier_normal(p.weight)

    # loss
    criterion = OnlineTripletLoss(margin=loss_margin,
                                  triplet_selector=SemihardNegativeTripletSelector(loss_margin, False))
    # criterion = OnlineTripletLoss(margin=loss_margin,triplet_selector=RandomNegativeTripletSelector(loss_margin, False))

    # optimizer
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9,
                    weight_decay=0.0005)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # save best
    best_loss = 1.0
    best_map = 0.0
    is_best = False

    # set model
    log.info('>> Setting Model ...')
    ckpts = None  # './ckpts/rmac-rdng/2d-sigmoid/freeze/lr-0.001-margin-0.5/model_epoch_0.pth.tar'
    if ckpts:
        log.info('>> Load Model ... {}'.format(ckpts))
        ckpts = torch.load(ckpts)
        start_epoch = ckpts['epoch']
        model.load_state_dict(ckpts['model_state_dict'])
        optimizer.load_state_dict(ckpts['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        # scheduler.last_epoch = start_epoch - 1
        it_train = ckpts['iterators'][0]
        it_valid = ckpts['iterators'][1]
        model.cuda()
        model = nn.DataParallel(model)
    else:
        model.cuda()
        model = nn.DataParallel(model)
        save_ckpt({
            'epoch': 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iterators': [it_train, it_valid],
        }, is_best, ckpt_path)

    # visualize
    v_dt = ListDataSet(rparis.get_visulize_list(n_visualize_cls), image_size=1024, bbx=None, transform=test_trn)
    showAtmp(v_dt, model.module.cuda(), visualize_path, 0)
    # showAtmp_3D_ver2(v_dt, model.module.cuda(), visualize_path, 0)
    # ch = showAtmp_3D(v_dt, model.module.cuda(), visualize_path, 0)
    # log.info('>> visualize channel : {}'.format(ch))
    # base mAP

    '''
    log.info('===== test 224 ({})====='.format(roxford.name))
    test_prefetch(model, basic, roxford, 224, test_trn, '/AttentionRetrieval/prefetch/resnet50/224/roxford')
    log.info('===== test 224 ({})====='.format(rparis.name))
    test_prefetch(model, basic, rparis, 224, test_trn, '/AttentionRetrieval/prefetch/resnet50/224/rparis')
    log.info('===== test 224 ({})====='.format(oxford.name))
    test_prefetch(model, basic, oxford, 224, test_trn, '/AttentionRetrieval/prefetch/resnet50/224/oxford')
    log.info('===== test 224 ({})====='.format(paris.name))
    test_prefetch(model, basic, paris, 224, test_trn, '/data/prefetch/resnet50/224/paris')

    test(model, basic, paris, 224, test_trn)
    '''
    log.info('===== Base model 224 ({})====='.format(paris.name))
    #map = test_basic(basic,paris,224,test_trn,None)

    log.info('===== test 224 ({}/{}) ====='.format(paris.name, oxford.name))
    # map = test(model, basic, paris, 224, test_trn,None)
    # map = test_basic(model,paris,224,test_trn,None)
    #map = test_prefetch(model, basic, preattn, prebasic, paris, 224, test_trn, None,
    #                    '/AttentionRetrieval/prefetch/base4/224/paris')
    #map = test_prefetch(model, basic, preattn, prebasic, oxford, 224, test_trn, None,
    #                    '/AttentionRetrieval/prefetch/base4/224/oxford')
    log.info('===== test 1024 ({}/{}) ====='.format(paris.name, oxford.name))
    # map = test(model, basic, paris, 1024, test_trn,None)
    #map = test_basic(model, oxford, 1024, test_trn,None)
    map = test_prefetch(model, basic, preattn, prebasic, paris, 1024, test_trn, None,
                        '/AttentionRetrieval/prefetch/base4/1024/paris')
    map = test_prefetch(model, basic, preattn, prebasic, oxford, 1024, test_trn, None,
                        '/AttentionRetrieval/prefetch/base4/1024/oxford')
    for ep in range(start_epoch, Epoch + 1):
        is_best = False
        # scheduler.step()
        # train
        loss, dp, dn, n_tr, it_train = train(model, trainLoader, optimizer, criterion, ep, it_train, verbose=verbose)

        # valid
        loss, dp, dn, n_tr, it_valid, whiten = valid(model, validLoader, criterion, ep, it_valid, verbose=verbose)

        # visualize
        showAtmp(v_dt, model.module.cuda(), visualize_path, ep)
        # showAtmp_3D_ver2(v_dt, model.module.cuda(), visualize_path, ep)
        # showAtmp_3D(v_dt, model.module.cuda(), visualize_path, ep, ch)
        # test
        log.info('===== test 224 ({}/{}) ====='.format(paris.name,oxford.name))
        #map = test(model, basic, paris, 224, test_trn,whiten)
        #map = test_basic(model, paris, 224, test_trn,whiten)
        #map = test_prefetch(model, basic, preattn, prebasic, paris, 224, test_trn, whiten,
        #                    '/AttentionRetrieval/prefetch/base4/224/paris')
        #map = test_prefetch(model, basic, preattn, prebasic, oxford, 224, test_trn, whiten,
        #                    '/AttentionRetrieval/prefetch/base4/224/oxford')

        if ep % 1 == 0:
            log.info('===== test 1024 ({}/{}) ====='.format(paris.name,oxford.name))
            #map = test(model, basic, paris, 1024, test_trn,whiten)
            #map = test_basic(model, oxford, 1024, test_trn,whiten)
            map = test_prefetch(model, basic, preattn, prebasic, paris, 1024, test_trn, whiten,
                                '/AttentionRetrieval/prefetch/base4/1024/paris')
            map = test_prefetch(model, basic, preattn, prebasic, oxford, 1024, test_trn, whiten,
                                '/AttentionRetrieval/prefetch/base4/1024/oxford')

        if map > best_map:
            best_loss = loss
            best_map = map
            is_best = True

        save_ckpt({
            'epoch': ep + 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iterators': [it_train, it_valid],
        }, is_best, ckpt_path)

    msg = 'Training End with\nloss - {}\tlr - {}\tmargin - {}\tbatch size - {}\tckpts - {}\n\nBest Map - {}\n\n\n\n' \
        .format(method, lr, loss_margin, batch_size, ckpt_path, best_map)
    log.error(msg)
