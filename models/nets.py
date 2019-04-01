from torchvision.models import resnet50, vgg16,resnet101
from torch.autograd import variable as V
from torch.utils import model_zoo
from torch.nn import functional as F
from torch import nn
import torch

resnet50_model_zoo_url = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth'
resnet101_model_zoo_url = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth'


class Attn(nn.Module):
    def __init__(self, attention, pool):
        super(Attn, self).__init__()
        self.base = nn.Sequential(*list(resnet50(pretrained=False).children())[:-2])
        self.base.load_state_dict(model_zoo.load_url(resnet50_model_zoo_url))
        self.attention = attention
        self.pool = pool

    def forward(self, x):
        base_out = self.base(x)
        attention_out, attention_map = self.attention(base_out)
        out = self.pool(attention_out)
        norm_out = F.normalize(out, p=2, dim=1)
        return norm_out, attention_map


class Basic(nn.Module):
    def __init__(self, pool):
        super(Basic, self).__init__()
        self.base = nn.Sequential(*list(resnet50(pretrained=False).children())[:-2])
        self.base.load_state_dict(model_zoo.load_url(resnet50_model_zoo_url))
        self.pool = pool

    def forward(self, x):
        base_out = self.base(x)
        out = self.pool(base_out)
        norm_out = F.normalize(out, p=2, dim=1)
        attention_map = F.avg_pool3d(base_out, (base_out.size(1), 1, 1))
        return norm_out, attention_map


if __name__ == '__main__':
    import torchvision.transforms as trn
    from PIL import Image

    from models.pooling import *
    from attention import *

    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    im = Image.open('/data/paris6k/jpg/paris_defense_000605.jpg')
    im = im.convert('RGB')
    im = V(transform(im)).unsqueeze(0)

    base = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])

    basic = Basic(RMAC())
    # att2d = Attention2D()
    # att3d = Attention3D()
    x = basic(im)
    print(x[0])
