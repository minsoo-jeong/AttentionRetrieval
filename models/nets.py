from torchvision.models import resnet50, vgg16, resnet101
from torch.autograd import variable as V
from torch.utils import model_zoo
from torch.nn import functional as F
from torch import nn
import torch

import numpy as np
import os

resnet50_model_zoo_url = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth'
resnet101_model_zoo_url = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth'
resnet101_gem_best_url = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth'

'''
res101_best = model_zoo.load_url(resnet101_gem_best_url)['state_dict']

from collections import OrderedDict

ndict = OrderedDict()

for k in res101_best.keys():
    l = k.split('.')
    if l[0] == 'features':
        nl = '.'.join(l[1:])
        ndict[nl] = res101_best[k]

torch.save(ndict, 'retrievalSfM120k-resnet101.pth')
'''


class L2N(nn.Module):
    def __init__(self, eps=1e-12):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


class Attn(nn.Module):
    def __init__(self, attention, pool):
        super(Attn, self).__init__()
        self.base = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.base.load_state_dict(model_zoo.load_url(resnet50_model_zoo_url))
        # self.base.load_state_dict(torch.load('retrievalSfM120k-resnet101.pth'))
        self.attention = attention
        self.pool = pool
        self.norm = L2N()

    def forward(self, x):
        base_out = self.base(x)
        attention_out, attention_map = self.attention(base_out)
        out = self.pool(attention_out)
        norm_out = self.norm(out)
        return norm_out, attention_map #,attention_out


class Prefetch_Attn(nn.Module):
    def __init__(self, attention, pool):
        super(Prefetch_Attn, self).__init__()
        self.attention = attention
        self.pool = pool
        self.norm = L2N()

    def forward(self, x):
        attention_out, attention_map = self.attention(x)
        out = self.pool(attention_out)
        norm_out = self.norm(out)
        return norm_out, attention_map


class Prefetch_Basic(nn.Module):
    def __init__(self, pool):
        super(Prefetch_Basic, self).__init__()
        self.pool = pool
        self.norm = L2N()

    def forward(self, x):
        out = self.pool(x)
        norm_out = self.norm(out)
        attention_map = F.avg_pool3d(x, (x.size(1), 1, 1))
        return norm_out, attention_map


class Basic(nn.Module):
    def __init__(self, pool):
        super(Basic, self).__init__()
        self.base = nn.Sequential(*list(resnet50(pretrained=False).children())[:-2])
        self.base.load_state_dict(model_zoo.load_url(resnet50_model_zoo_url))
        # self.base.load_state_dict(torch.load('retrievalSfM120k-resnet101.pth'))
        self.pool = pool
        self.norm = L2N()

    def forward(self, x):
        base_out = self.base(x)
        out = self.pool(base_out)
        norm_out = self.norm(out)
        attention_map = F.avg_pool3d(base_out, (base_out.size(1), 1, 1))
        return norm_out, attention_map


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.base = nn.Sequential(*list(resnet50(pretrained=False).children())[:-2])
        self.base.load_state_dict(model_zoo.load_url(resnet50_model_zoo_url))

    def forward(self, x):
        base_out = self.base(x)
        return base_out


def extract_vectors(model, loader, path):
    if not os.path.exists(path):
        os.makedirs(path)

    print('extract vector...', end='')
    model.eval()
    with torch.no_grad():
        for n, i in enumerate(loader):
            out = model(V(i[0]).cuda())
            torch.save(out.cpu().data, os.path.join(path, '{}.pth.tar'.format(str(n).zfill(4))))
            if (n + 1) % 50 == 0 or (n + 1) == len(loader.dataset):
                print('\r>>>> {}/{} done...'.format(n * loader.batch_size + out.size(0), len(loader.dataset)), end='')
        print('\r', end='')


def extract():
    from test import GroundTruth
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.curdir))))
    sys.path.append(os.path.abspath(os.path.curdir))
    from Network import dataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as trn
    import attention
    import pooling
    import shutil

    ckpts = '/data/ckpts/RMAC/base4/lr-0.001-margin-0.2-rndm256/model_epoch_16.pth.tar'
    #trained = Attn(attention.AttentionCS(pool=pooling.GeM()), pooling.GeM())
    #trained.load_state_dict(torch.load(ckpts)['model_state_dict'])

    trained=Basic(pooling.RMAC())
    trained.load_state_dict(torch.load(ckpts)['model_state_dict'])

    model = Base()
    model.base.load_state_dict(trained.base.state_dict())

    model.cuda()
    model = nn.DataParallel(model)
    rparis = GroundTruth(os.path.join('/data', 'rparis6k', 'jpg'),
                         os.path.join('/data', 'rparis6k', 'gnd_rparis6k.pkl'))
    roxford = GroundTruth(os.path.join('/data', 'roxford5k', 'jpg'),
                          os.path.join('/data', 'roxford5k', 'gnd_roxford5k.pkl'))
    paris = GroundTruth(os.path.join('/paris6k', 'jpg'),
                        os.path.join('/paris6k', 'gnd_paris6k.pkl'))
    oxford = GroundTruth(os.path.join('/oxford5k', 'jpg'),
                         os.path.join('/oxford5k', 'gnd_oxford5k.pkl'))

    test_trn = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if not os.path.exists('/data/prefetch/base4/224/oxford'):
        os.makedirs('/data/prefetch/base4/224/oxford')
    if not os.path.exists('/data/prefetch/base4/1024/oxford'):
        os.makedirs('/data/prefetch/base4/1024/oxford')
    if not os.path.exists('/data/prefetch/base4/224/paris'):
        os.makedirs('/data/prefetch/base4/224/paris')
    if not os.path.exists('/data/prefetch/base4/1024/paris'):
        os.makedirs('/data/prefetch/base4/1024/paris')

    loader = DataLoader(dataset.ListDataSet(oxford.real_imlist, image_size=224, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=4)
    extract_vectors(model, loader, '/AttentionRetrieval/prefetch/base4/224/oxford')
    print('224-oxford')

    loader = DataLoader(dataset.ListDataSet(oxford.real_imlist, image_size=1024, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=4)
    extract_vectors(model, loader, '/AttentionRetrieval/prefetch/base4/1024/oxford')
    print('1024-oxford')

    loader = DataLoader(dataset.ListDataSet(paris.real_imlist, image_size=224, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=4)
    extract_vectors(model, loader, '/AttentionRetrieval/prefetch/base4/224/paris')
    print('224-paris')

    loader = DataLoader(dataset.ListDataSet(paris.real_imlist, image_size=1024, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=4)
    extract_vectors(model, loader, '/AttentionRetrieval/prefetch/base4/paris')
    print('1024-paris')


    '''
    loader = DataLoader(dataset.ListDataSet(rparis.real_imlist, image_size=224, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=8)
    extract_vectors(model, loader, '/data/prefetch/resnet50-rmac-base-8024/224/rparis')

    loader = DataLoader(dataset.ListDataSet(roxford.real_imlist, image_size=224, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=8)
    extract_vectors(model, loader, '/data/prefetch/resnet50-rmac-base-8024/224/roxford')

    loader = DataLoader(dataset.ListDataSet(oxford.real_imlist, image_size=224, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=8)
    extract_vectors(model, loader, '/data/prefetch/resnet50-rmac-base-8024/224/oxford')

    loader = DataLoader(dataset.ListDataSet(rparis.real_imlist, image_size=1024, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=8)
    extract_vectors(model, loader, '/data/prefetch/resnet50-rmac-base-8024/1024/rparis')


    loader = DataLoader(dataset.ListDataSet(roxford.real_imlist, image_size=1024, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=8)
    extract_vectors(model, loader, '/data/prefetch/resnet50-rmac-base-8024/1024/roxford')

    loader = DataLoader(dataset.ListDataSet(oxford.real_imlist, image_size=1024, bbx=None, transform=test_trn),
                        batch_size=1, shuffle=False, num_workers=8)
    extract_vectors(model, loader, '/data/prefetch/resnet50-rmac-base-8024/1024/oxford')
    '''

if __name__ == '__main__':
    extract()
    '''
    print()
    import torchvision.transforms as trn
    from PIL import Image

    from models.pooling import *
    from models.attention import *

    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    im = Image.open('/data/paris6k/jpg/paris_defense_000605.jpg')
    im = im.convert('RGB')
    im = V(transform(im)).unsqueeze(0)

    # a=Attn(Attention2D(),GeM())

    base = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
    print(base.state_dict()['1.running_mean'])
    print(base.state_dict()['0.weight'][0][0])
    basic = Basic(RMAC())
    print(basic.state_dict().keys())
    print(basic.base.state_dict()['1.running_mean'])
    print(basic.base.state_dict()['0.weight'][0][0][0])
    # att2d = Attention2D()
    # att3d = Attention3D()
    x = basic(im)
    print(x[0])

    attn = Attn(Attention2D(), RMAC())
    print(attn.state_dict().keys())
    print(base.state_dict()['1.running_mean'][:5])
    print(basic.base.state_dict()['1.running_mean'][:5])
    print(base.state_dict()['0.weight'][0][0][0])
    print(basic.base.state_dict()['0.weight'][0][0][0])
    '''
