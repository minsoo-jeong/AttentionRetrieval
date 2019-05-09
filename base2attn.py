import torch
from models.nets import *
from models.pooling import *
from models.attention import *


src = '/data/ckpts/RMAC/base/lr-0.001-margin-0.2/model_epoch_12.pth.tar'

base=Basic(RMAC())
model = Attn(AttentionCS(pool=RMAC()), RMAC())

ckpt=torch.load(src)
base.load_state_dict(ckpt['model_state_dict'])
model.base.load_state_dict(base.base.state_dict())

torch.save(model.state_dict(),'./RMAC-base-8118.pth.tar')

model = Attn(AttentionCS(pool=RMAC()), RMAC())
model.load_state_dict(torch.load('./RMAC-base-8118.pth.tar'))
print(model)