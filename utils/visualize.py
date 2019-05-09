import torchvision.transforms as trn
from torch.utils.data import DataLoader
import torch.autograd.variable as V
from torch import nn
import numpy as np
import cv2
import os
import torch
from PIL import Image


def visualize(img, att_map, path, idx, ep):
    shape = img[0].size()
    up = nn.Upsample(size=(shape[-2], shape[-1]), scale_factor=None, mode='bilinear', align_corners=True)
    heat = up(att_map).squeeze(0).squeeze(0).detach().cpu().numpy()

    heat = heat * 255.
    _, heat = cv2.threshold(heat, 255., 255., cv2.THRESH_TRUNC)

    cv_heat = np.uint8(cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR))
    cv_heat = cv2.applyColorMap(cv_heat, cv2.COLORMAP_JET)

    pil_image = cv2.cvtColor(cv_heat, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(pil_image)
    im_pil.thumbnail((224,224), Image.ANTIALIAS)
    numpy_image = np.array(im_pil)
    cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(path, '{}-heatmap-ep{}.jpg'.format(idx, ep)), cv_image)

    toPIL = trn.Compose([
        trn.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                      std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
                      inplace=False),
        trn.ToPILImage(mode='RGB')
    ])
    origin = toPIL(img[0])
    origin = np.array(origin)
    origin = cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./visual/{}_{}_ori.jpg'.format(name, str(epoch), str(iter)), origin)

    blend = cv2.addWeighted(origin, 0.6, cv_heat, 0.4, 0)

    pil_image = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(pil_image)
    im_pil.thumbnail((224,224), Image.ANTIALIAS)
    numpy_image = np.array(im_pil)
    cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(path, '{}-ep{}.jpg'.format(idx, ep)), cv_image)


def visualize_3D(img, att_map, path, idx, ep, ch):
    shape = img[0].size()
    toPIL = trn.Compose([
        trn.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                      std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
                      inplace=False),
        trn.ToPILImage(mode='RGB')
    ])
    for n, c in enumerate(ch):
        att_map_c = att_map[:, c:c + 1, :, :]
        up = nn.Upsample(size=(shape[-2], shape[-1]), scale_factor=None, mode='bilinear', align_corners=True)
        heat = up(att_map_c).squeeze(0).squeeze(0).detach().cpu().numpy()

        heat = heat * 255.
        _, heat = cv2.threshold(heat, 255., 255., cv2.THRESH_TRUNC)

        cv_heat = np.uint8(cv2.cvtColor(heat, cv2.COLOR_GRAY2BGR))
        cv_heat = cv2.applyColorMap(cv_heat, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(path, '{}-ch{}-heatmap-ep{}.jpg'.format(idx, str(n), ep)), cv_heat)

        origin = toPIL(img[0])
        origin = np.array(origin)
        origin = cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./visual/{}_{}_ori.jpg'.format(name, str(epoch), str(iter)), origin)

        blend = cv2.addWeighted(origin, 0.6, cv_heat, 0.4, 0)
        cv2.imwrite(os.path.join(path, '{}-ch{}-ep{}.jpg'.format(idx, str(n), ep)), blend)

def showAtmp_3D(dataset, model, path, ep, ch=None):
    if not os.path.exists(path):
        os.makedirs(path)
    model.eval()
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=1)
        for n, i in enumerate(loader):
            if ep == 0:
                ori = cv2.imread(i[1][0])
                shape = i[0].size()
                ori = cv2.resize(ori, (shape[-1], shape[-2]))
                cv2.imwrite(os.path.join(path, '{}.jpg'.format(str(n + 1))), ori)
            feature, attn_map = model(V(i[0]).cuda())

            if ch is None:
                _, max_channel_idx = torch.sort(feature, dim=1, descending=True)
                max_channel_idx = max_channel_idx.cpu().tolist()[0]
                ch = max_channel_idx[:2] + max_channel_idx[1023:1025] + max_channel_idx[2046:]

            visualize_3D(i[0], attn_map, path, str(n + 1), str(ep), ch)

        return ch


def showAtmp(dataset, model, path, ep):
    if not os.path.exists(path):
        os.makedirs(path)
    model.eval()
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=1)
        for n, i in enumerate(loader):
            if ep == 0:
                ori = cv2.imread(i[1][0])
                cv2.imwrite(os.path.join(path, '{}.jpg'.format(str(n + 1))), ori)
            feature, attn_map = model(V(i[0]).cuda())
            visualize(i[0], attn_map, path, str(n + 1), str(ep))

def showAtmp_3D_ver2(dataset, model, path, ep):
    if not os.path.exists(path):
        os.makedirs(path)
    model.eval()
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=1)
        in_class_features = []
        in_class_attentions = []
        in_class_imgs = []
        for n, i in enumerate(loader):
            if ep == 0:
                ori = cv2.imread(i[1][0])
                shape = i[0].size()
                ori = cv2.resize(ori, (shape[-1], shape[-2]))
                cv2.imwrite(os.path.join(path, '{}.jpg'.format(str(n + 1))), ori)
            feature, attn_map = model(V(i[0]).cuda())
            in_class_features.append(feature.data.cpu().numpy())
            in_class_attentions.append(attn_map.data.cpu())
            in_class_imgs.append(i[0])
            if n % 4 == 3:
                q_feature = in_class_features[0]
                ch = []
                for j in range(1, 4):
                    min_ch = list(np.argsort(abs(q_feature - in_class_features[j]))[0])
                    ch += min_ch[:2]
                for j in range(0, 4):
                    visualize_3D(in_class_imgs[j], in_class_attentions[j], path, str(n - 2 + j), str(ep), ch)
                in_class_features = []
                in_class_attentions = []
                in_class_imgs = []

        return ch

