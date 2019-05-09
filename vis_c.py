from torchvision.transforms import transforms as trn
from torch.autograd import Variable as V

from models import nets
from models.pooling import *
from models.attention import *
from test import GroundTruth
from Network import dataset

import numpy as np
import os
import cv2
from PIL import Image

a = Image.open('2.jpg')
case = 2
case=str(case).zfill(3)
origin = np.array(a)
origin = cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
# Test DataSet
test_trn = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

paris = GroundTruth(os.path.join('/paris6k', 'jpg'),
                    os.path.join('/paris6k', 'gnd_paris6k.pkl'))

oxford = GroundTruth(os.path.join('/oxford5k', 'jpg'),
                     os.path.join('/oxford5k', 'gnd_oxford5k.pkl'))

pool = RMAC()

trained = nets.Basic(pool)
trained.load_state_dict(
    torch.load('/data/ckpts/RMAC/base4/lr-0.001-margin-0.2-rndm256/model_epoch_16.pth.tar')['model_state_dict'])

model = nets.Base()
model.base.load_state_dict(trained.base.state_dict())
im = dataset.ListDataSet(paris.real_imlist, image_size=1024, bbx=None, transform=test_trn)
im = dataset.ListDataSet(paris.get_visulize_list(5), image_size=1024, bbx=None, transform=test_trn)
im = test_trn(a)

# img=im[0][0]
img = im
print(img.unsqueeze(0).shape)

c = model(V(img).unsqueeze(0))
print(c.shape)
c = F.avg_pool3d(c, (c.size(1), 1, 1))

attn = nets.Attn(AttentionCS(pool), pool)
attn.load_state_dict(
    torch.load(
        '/data/ckpts/RMAC/3D-cs/Cpool-rmac-Sigmoid/freeze9/lr-0.001-margin-0.2-rndm280/model_epoch_2.pth.tar')[
        'model_state_dict'])

_, am, ao = attn(V(img).unsqueeze(0))

ao = F.avg_pool3d(ao, (ao.size(1), 1, 1))

up = nn.Upsample(size=(img.shape[-2], img.shape[-1]), scale_factor=None, mode='bilinear',
                 align_corners=True)

am = am * 255.
am = up(am).squeeze(0).squeeze(0).detach().numpy()
_, am = cv2.threshold(am, 255., 255., cv2.THRESH_TRUNC)
cv_am = np.uint8(cv2.cvtColor(am, cv2.COLOR_GRAY2BGR))
cv_am = cv2.applyColorMap(cv_am, cv2.COLORMAP_JET)
print(am.shape)
print(np.max(am))
cv2.imwrite('channel/case-%s-attentionmap.jpg' % case, cv_am)
blend = cv2.addWeighted(origin, 0.6, cv_am, 0.4, 0)
cv2.imwrite('channel/case-%s-attentionmap-blend.jpg' % case, blend)

c = c * 255.
c = up(c).squeeze(0).squeeze(0).detach().numpy()
_, c = cv2.threshold(c, 255., 255., cv2.THRESH_TRUNC)
cvc = np.uint8(cv2.cvtColor(c, cv2.COLOR_GRAY2BGR))
cvc = cv2.applyColorMap(cvc, cv2.COLORMAP_JET)
print(cvc.shape)
print(np.max(c))
cv2.imwrite('channel/case-%s-base.jpg' % case, cvc)
blend = cv2.addWeighted(origin, 0.6, cvc, 0.4, 0)
cv2.imwrite('channel/case-%s-base-blend.jpg' % case, blend)

ao = ao * 255.
ao = up(ao).squeeze(0).squeeze(0).detach().numpy()
_, ao = cv2.threshold(ao, 255., 255., cv2.THRESH_TRUNC)
cvao = np.uint8(cv2.cvtColor(ao, cv2.COLOR_GRAY2BGR))
cvao = cv2.applyColorMap(cvao, cv2.COLORMAP_JET)
print(cvc.shape)
print(np.max(c))
cv2.imwrite('channel/case-%s-attn.jpg' % case, cvao)
blend = cv2.addWeighted(origin, 0.6, cvao, 0.4, 0)
cv2.imwrite('channel/case-%s-attn-blend.jpg' % case, blend)


def show_channel(tensor, postfix, ch, shape):
    n_ch = tensor.size(1)
    print(n_ch)
    up = nn.Upsample(size=shape, scale_factor=None, mode='bilinear',
                     align_corners=True)

    for c in ch:
        map = tensor[0, c]
        map = map * 255.
        map = up(map.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy()

        _, map = cv2.threshold(map, 255., 255., cv2.THRESH_TRUNC)
        cv_map = np.uint8(cv2.cvtColor(map, cv2.COLOR_GRAY2BGR))
        cv_map = cv2.applyColorMap(cv_map, cv2.COLORMAP_JET)
        print(map.shape)
        print(np.max(map))
        cv2.imwrite('channel/imc-{}-{}.jpg'.format(c, postfix), cv_map)

# ch = np.random.choice(2048, 10, False)

# show_channel(c, 'base', ch,(im[0][0].shape[-2], im[0][0].shape[-1]))
# show_channel(ao, 'attn', ch,(im[0][0].shape[-2], im[0][0].shape[-1]))
