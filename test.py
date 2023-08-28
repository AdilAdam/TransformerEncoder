import json
import os

import torch
from models.embedding import Embedding, Posionalencoding

from models.attentions.attention import DCTAttention, ScaledDotProductAttention

emb = Embedding(vocab_size=16, d_model=128)
pos_emb = Posionalencoding(d_model=128)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cuda") if torch.cuda.is_available else torch.device("cuda")
device = torch.device("cpu")
yaml_file = "/home/junlin/myproject/MyMOdel/config/config.json"
with open(yaml_file, "r") as cnfg:
    confg = json.load(cnfg)

model_1 = ScaledDotProductAttention()
model_ = DCTAttention(cnf=confg)
data = "./data.json"
f = open(data)
data = json.load(f)

train_data = data["data"]["train"]
train_data = data["data"]["train"]
label = data["data"]["label"]
cv_data = data["data"]["valid"]
cv_label = data["data"]["v_label"]
data = torch.tensor(train_data).to(device)
label = torch.tensor(label).to(device)
cv_data = torch.tensor(cv_data).to(device)
cv_label = torch.tensor(cv_label).to(device)

emb.to(device)
pos_emb.to(device)
model_.to(device)

rslt = pos_emb(emb(data))
# print(rslt.shape)


import numpy as np
from torch.autograd import Variable


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    np_mask = Variable(torch.from_numpy(np_mask == 1))
    return np_mask


def create_masks(src, trg):
    src_mask = (src != 1).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != 1).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


src_mask, trg_mask = create_masks(data, label)
# print(src_mask.shape)
a, b = model_(Q=rslt, K=rslt, V=rslt, mask=src_mask.squeeze(1))

c, d = model_1(Q=rslt, K=rslt, V=rslt, mask=src_mask.squeeze(1))

