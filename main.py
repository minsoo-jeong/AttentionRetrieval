import torchvision.transforms as trn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim import SGD
import torch.nn as nn

from datetime import datetime
import warnings
import os

from models import nets, pooling, attention
from Network import losses, Network, dataset
from test import GroundTruth, test
from train import *

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    Epoch = 50
    start_epoch = 1
    it_train = 0
    it_valid = 0
    # dataset
    print('>> Dataset ... ')
    # train / valid DataSet
    batch_size = 16
    transform_valid = transform_train = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainSet = dataset.TripletDataset(os.path.join('/data', 'landmark', 'Landmark-clean', 'train'),
                                      transform=transform_train)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=8)

    validSet = dataset.TripletDataset(os.path.join('/data', 'landmark', 'Landmark-clean', 'valid'),
                                      transform=transform_valid)
    validLoader = DataLoader(validSet, batch_size=50, shuffle=True, num_workers=8)

    # Test DataSet
    test_trn = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rparis = GroundTruth(os.path.join('/data', 'rparis6k', 'jpg'),
                         os.path.join('/data', 'rparis6k', 'gnd_rparis6k.pkl'))

    roxford = GroundTruth(os.path.join('/data', 'roxford5k', 'jpg'),
                          os.path.join('/data', 'roxford5k', 'gnd_roxford5k.pkl'))

    paris = GroundTruth(os.path.join('/data', 'paris6k', 'jpg'),
                        os.path.join('/data', 'paris6k', 'gnd_paris6k.pkl'))

    oxford = GroundTruth(os.path.join('/data', 'oxford5k', 'jpg'),
                         os.path.join('/data', 'oxford5k', 'gnd_oxford5k.pkl'))

    # make model
    print('>> Create Model ... rmac-attention2D')
    # embed = nets.Basic(pooling.RMAC())
    embed = nets.Attn(attention.Attention2D(), pooling.RMAC())
    model = Network.TripletNet(embed)
    # freeze backbone
    print('freeze backbone / init attention')
    for n, p in model.embedding_net.base.named_parameters():
        p.requires_grad = False

    # initialize attention
    for n, p in model.embedding_net.attention.named_modules():
        if isinstance(p, torch.nn.Conv2d):
            nn.init.xavier_normal(p.weight)

    # loss
    criterion = losses.TripletLoss(margin=0.3)

    # optimizer
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # set model
    print('>> Setting Model ...')
    ckpts = None  # './ckpts/gem_attention2D/model_epoch_2.pth.tar'
    if ckpts:
        ckpts = torch.load(ckpts)
        start_epoch = ckpts['epoch']
        model.embedding_net.load_state_dict(ckpts['model_state_dict'])
        optimizer.load_state_dict(ckpts['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        scheduler.last_epoch = start_epoch - 1
        it_train = ckpts['iterators'][0]
        it_valid = ckpts['iterators'][1]
    model.cuda()
    model = nn.DataParallel(model)

    # save best
    best_loss = 1.0
    best_pos = 1.0
    best_neg = 0.0
    best_map = 0.0
    is_best = False

    # base mAP
    test_model = model.module.embedding_net
    test_model = nn.DataParallel(test_model)
    test(test_model, paris, 224, test_trn)
    test(test_model, paris, 1024, test_trn)

    for ep in range(start_epoch, Epoch + 1):
        is_best = False
        scheduler.step()
        # train
        loss, dp, dn, it_train = train(model, trainLoader, optimizer, criterion, ep, it_train, verbose=False)

        # valid
        loss, dp, dn, it_valid = valid(model, validLoader, criterion, ep, it_valid, verbose=False)

        # test
        test_model = model.module.embedding_net
        test_model = nn.DataParallel(test_model)
        test(test_model, paris, 224, test_trn)
        if ep % 5 == 0:
            map = test(test_model, paris, 1024, test_trn)
            if map > best_map:
                best_loss = loss
                best_pos = dp
                best_neg = dn
                is_best = True

            save_ckpt({
                'epoch': ep + 1,
                'model_state_dict': model.module.embedding_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iterators': [it_train, it_valid],
            }, is_best, './ckpts/rmac_attention2D_freeze_01/')
