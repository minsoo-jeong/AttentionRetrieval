import torchvision.transforms as trn
from torch import nn
import numpy as np
import cv2
import os


# check
def visualize(img, att_map, path):
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    dir = os.path.dirname(path)
    fullpath_heat=os.path.join(dir,'{}_heat.{}'.format(name,ext))
    fullpath = os.path.join(dir, '{}.{}'.format(name, ext))

    shape = img.size()
    up = nn.Upsample(size=(shape[-2], shape[-1]), scale_factor=None, mode='bilinear', align_corners=True)
    heat = up(att_map).squeeze(0).squeeze(0).detach().cpu().numpy()
    heat = heat * 255 / (np.max(heat) + 1e-12)

    cv_heat = np.uint8(cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR))
    cv_heat = cv2.applyColorMap(cv_heat, cv2.COLORMAP_JET)
    # cv2.imwrite('./visual/{}_{}_heat.jpg'.format(name,str(epoch), str(iter)), cv_heat)
    cv2.imwrite('./visual/{}_{}_heat.jpg'.format(name, str(epoch), str(iter)), cv_heat)
    toPIL = trn.Compose([
        trn.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                      std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
                      inplace=False),
        trn.ToPILImage(mode='RGB')
    ])
    origin = toPIL(img[0])
    origin = np.array(origin)
    origin = cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('./visual/{}_{}_ori.jpg'.format(name, str(epoch), str(iter)), origin)

    blend = cv2.addWeighted(origin, 0.6, cv_heat, 0.4, 0)
    cv2.imwrite('./visual/{}_{}_{}.jpg'.format(name, str(epoch), str(iter)), blend)
