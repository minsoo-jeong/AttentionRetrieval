import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch


class MAC(nn.Module):
    def __init__(self):
        super(MAC, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(7, 7))

    def forward(self, feature):
        mac = self.pool(feature)
        return mac.squeeze(-1).squeeze(-1)


class SPoC(nn.Module):
    def __init__(self):
        super(SPoC, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=(7, 7))

    def forward(self, feature):
        spoc = self.pool(feature)
        return spoc.squeeze(-1).squeeze(-1)


class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-12):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        rmac = self.rmac(x, L=self.L, eps=self.eps)
        return rmac.squeeze(-1).squeeze(-1)

    def rmac(self, x, L=3, eps=1e-12):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

        for l in range(1, L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                    R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                    v += vt

        return v


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-12):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        gem = self.gem(x, p=self.p, eps=self.eps)
        return gem.squeeze(-1).squeeze(-1)

    def gem(self, x, p=3, eps=1e-12):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


if __name__=='__main__':
    import numpy as np
    import torchvision.transforms as trn

    a=torch.Tensor(np.arange(2*5*6*6).reshape(2,5,6,6))/2*5*6*6
    b=trn.Compose(trn.Resize(3))
    c=torch.nn.functional.interpolate(a,scale_factor=0.5,mode='bilinear')
    #c=b(a)
    print(a[0,1])
    print(c[0,1],c.size())
    g = np.random.choice(5, 1)
    print(g)