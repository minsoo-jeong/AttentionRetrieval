import torch.nn.functional as F
import torch.nn as nn
import torch


class Attention2D(nn.Module):
    def __init__(self, input_channel=2048):
        super(Attention2D, self).__init__()
        self.attention_score = nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channel, out_channels=512, kernel_size=(1, 1), bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1), bias=False)
        )
        self.attention_prob = torch.nn.Softplus()

    def forward(self, x):
        attention_score = self.attention_score(x)
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
            torch.nn.Conv2d(in_channels=input_channel, out_channels=512, kernel_size=(1, 1), bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=input_channel, kernel_size=(1, 1), bias=False)
        )
        self.attention_prob = torch.nn.Softplus()

    def forward(self, x):
        attention_score = self.attention_score(x)
        attention_prob = self.attention_prob(attention_score)

        attention_x = x * attention_prob
        return attention_x, attention_prob
