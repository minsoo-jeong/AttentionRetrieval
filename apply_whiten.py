import numpy as np
from utils.whiten import *

from test import *

ox = np.load('out/oxford.npy')
pa = np.load('out/paris.npy')

ox_q = np.load('out/oxford_q.npy')
pa_q = np.load('out/paris_q.npy')

'''
ox_m, ox_p = pcawhitenlearn(ox)
print('pca -ox')
pa_m, pa_p = pcawhitenlearn(pa)
print('pca -pa')

ox_w = whitenapply(ox, pa_m, pa_p)
oxq_w = whitenapply(ox_q, pa_m, pa_p)

pa_w = whitenapply(pa, ox_m, ox_p)
paq_w = whitenapply(pa_q, ox_m, ox_p)
'''
ox_u,ox_s,ox_v,ox_m=learningPCA2(ox)
print('pca -ox')
pa_u,pa_s,pa_v,pa_m=learningPCA2(pa)
print('pca -pa')
ox_w=apply_whitening2(ox,pa_m,pa_u,pa_s)
oxq_w=apply_whitening2(ox_q,pa_m,pa_u,pa_s)
pa_w=apply_whitening2(pa,ox_m,ox_u,ox_s)
paq_w=apply_whitening2(pa_q,ox_m,ox_u,ox_s)


paris = GroundTruth(os.path.join('/paris6k', 'jpg'),
                    os.path.join('/paris6k', 'gnd_paris6k.pkl'))

oxford = GroundTruth(os.path.join('/oxford5k', 'jpg'),
                     os.path.join('/oxford5k', 'gnd_oxford5k.pkl'))

score = np.dot(ox_q, ox.T)
ranks = np.argsort(-score, axis=1)
map_whiten, msg = compute_map_and_print(ranks, oxford)
print('{} - q(Attn) : {}'.format(oxford.name, msg))


score = np.dot(oxq_w, ox_w.T)
ranks = np.argsort(-score, axis=1)
map_whiten, msg = compute_map_and_print(ranks, oxford)
print('{} - q(Attn) : {}'.format(oxford.name, msg))

score = np.dot(pa_q, pa.T)
ranks = np.argsort(-score, axis=1)
map_whiten, msg = compute_map_and_print(ranks, paris)
print('{} - q(Attn) : {}'.format(paris.name, msg))

score = np.dot(paq_w, pa_w.T)
ranks = np.argsort(-score, axis=1)
map_whiten, msg = compute_map_and_print(ranks, paris)
print('{} - q(Attn) : {}'.format(paris.name, msg))