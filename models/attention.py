import torch.nn.functional as F
import torch.nn as nn
import torch
import random


class L2N(nn.Module):

    def __init__(self, eps=1e-12):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


class Softmax2D(nn.Module):
    def __init__(self):
        super(Softmax2D, self).__init__()

    def forward(self, x):
        ori_shape = x.data.shape
        return F.softmax(x.view((ori_shape[0], ori_shape[1], ori_shape[2] * ori_shape[3])), dim=-1).view(ori_shape)


class Attention2D(nn.Module):
    def __init__(self, input_channel=2048):
        super(Attention2D, self).__init__()
        self.attention_score = nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channel, out_channels=512, kernel_size=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1))
        )
        # self.attention_prob = Softmax2D()
        # self.attention_prob = torch.nn.Softplus()
        self.attention_prob = torch.nn.Sigmoid()

    def forward(self, x):
        attention_score = self.attention_score(x)
        attention_prob = self.attention_prob(attention_score)

        attention_x = x * attention_prob

        return attention_x, attention_prob


class AttentionCS(nn.Module):
    def __init__(self, pool, input_channel=2048):
        super(AttentionCS, self).__init__()
        self.attention_score = nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channel, out_channels=512, kernel_size=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1))
        )
        self.attention_prob_c = torch.nn.Softmax()
        self.attention_prob = torch.nn.Sigmoid()  # Softmax2D() #torch.nn.Softmax()
        self.pool = pool

    def forward(self, x):
        attention_c_score = self.pool(x)
        # attention_c = self.attention_prob_c(attention_c_score).unsqueeze(-1).unsqueeze(-1)
        out_c = torch.mul(x, attention_c_score.unsqueeze(-1).unsqueeze(-1))
        attention_score = self.attention_score(out_c)
        attention_prob = self.attention_prob(attention_score)
        attention_x = x * attention_prob
        return attention_x, attention_prob


class AttentionCS2(nn.Module):
    def __init__(self, pool, input_channel=2048):
        super(AttentionCS2, self).__init__()
        self.attention_score = nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channel, out_channels=512, kernel_size=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1))
        )
        self.attention_prob_c = torch.nn.Softmax()
        self.attention_prob = torch.nn.Sigmoid()  # Softmax2D() #torch.nn.Softmax()
        self.pool = pool
        self.attention_score_c = nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        #channel = self.pool(x)
        channel=torch.nn.functional.avg_pool2d(x, (x.size(-2), x.size(-1))).squeeze(-1).squeeze(-1)
        #print(channel.shape)
        attention_c_score = self.attention_score_c(channel)
        # attention_c = self.attention_prob_c(attention_c_score).unsqueeze(-1).unsqueeze(-1)
        out_c = torch.mul(x, attention_c_score.unsqueeze(-1).unsqueeze(-1))
        attention_score = self.attention_score(out_c)
        attention_prob = self.attention_prob(attention_score)
        attention_x = x * attention_prob
        return attention_x, attention_prob


class Attention3D(nn.Module):
    def __init__(self, input_channel=2048):
        super(Attention3D, self).__init__()
        self.attention_wh_score = nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channel, out_channels=512, kernel_size=(1, 1), bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1), bias=False)
        )
        self.attention_c_score = nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.attention_prob = torch.nn.Softplus()

    def forward(self, x):
        att_score_hw = self.attention_wh_score(x)
        att_score_c = self.attention_c_score(x)

        attention_score = att_score_hw * att_score_c
        attention_prob = self.attention_prob(attention_score)
        attention_feature = x * attention_prob
        return attention_feature, attention_prob


class AttentionFCN(nn.Module):
    def __init__(self, input_channel=2048):
        super(AttentionFCN, self).__init__()
        self.attention_score = nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(1, 1)),
        )
        # self.attention_prob = torch.nn.Softplus()
        self.attention_prob = torch.nn.Sigmoid()

    def forward(self, x):
        attention_score = self.attention_score(x)
        attention_prob = self.attention_prob(attention_score)
        attention_x = x * attention_prob
        attention_x = attention_x + x
        return attention_x, attention_prob


class AttentionCh(nn.Module):
    def __init__(self, rescale=[1]):
        super(AttentionCh, self).__init__()
        self.attention_score = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048, bias=True),
            nn.BatchNorm2d(2048, eps=1e-8, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048, bias=True),
            nn.BatchNorm2d(2048, eps=1e-8, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048, bias=True),
            nn.BatchNorm2d(2048, eps=1e-8, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048, bias=True),
            nn.BatchNorm2d(2048, eps=1e-8, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048, bias=True),
            nn.BatchNorm2d(2048, eps=1e-8, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048, bias=True),
            nn.BatchNorm2d(2048, eps=1e-8, momentum=0.1, affine=True, track_running_stats=True),
        )
        # self.attention_prob = torch.nn.Softplus()
        self.attention_prob = torch.nn.Sigmoid()

        self.rescale = rescale

    def forward(self, x):
        h, w, = x.size(2), x.size(3)
        gamma = random.choice(self.rescale)
        new_h, new_w = int(h * gamma), int(w * gamma)
        x = F.upsample(x, size=(new_h, new_w), mode='bilinear')
        attention_score = self.attention_score(x)
        attention_prob = self.attention_prob(attention_score)
        attention_x = x * attention_prob
        # attention_x = attention_x+x
        return attention_x, attention_prob

    def change_rescale_gamma(self, gamma=[1.0]):
        self.rescale = gamma


class AttentionChannel(nn.Module):
    def __init__(self):
        super(AttentionChannel, self).__init__()
        self.attention_score = nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
        )
        # self.attention_prob = torch.nn.Softplus()
        self.attention_prob = torch.nn.Sigmoid()

    def forward(self, x):
        attention_score = self.attention_score(x.view((x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])))

        attention_score = attention_score.view(x.shape)
        attention_prob = self.attention_prob(attention_score)
        attention_x = x * attention_prob
        return attention_x, attention_prob


if __name__ == '__main__':
    import numpy as np
    from torch.autograd import Variable as V

    # t = np.arange(2 * 9 * 7 * 512.0).reshape(2, 512, 9, 7) / (2 * 512.)
    a = torch.Tensor(np.arange(2 * 9 * 2. * 2).reshape(2, 9, 2, 2))
    b = torch.Tensor(np.arange(1 * 1 * 2. * 2).reshape(1, 1, 2, 2))
    c = a * b
    print(a[0, 1], b[0, 0], c[0, 1])
