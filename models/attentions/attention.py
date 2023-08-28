#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, tempereture: float = None, drop_rate: float = 0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.pathsplt = nn.Conv2d(stride=2)
        self.d_k = tempereture
        self.drop_out = nn.Dropout(p=drop_rate)

    def forward(self, Q, K, V, mask=None) -> torch.Tensor:

        attn = torch.matmul(Q / self.d_k, K.transpose(-2, -1))

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 1, -1e9)
        attn = self.drop_out(F.softmax(attn, dim=-1))
        x = torch.matmul(attn, V)
        return x, attn


import sys
from math import sqrt

sys.path.append(".")
from utlis.torch_dct import create_dct, dct, idct


class DCTAttention(nn.Module):
    Q_dct = None

    def __init__(self, cnf):
        super().__init__()
        self.cnf = cnf

        # Attention stuff
        self.head_dim = cnf["attention"]["head_dim"]
        atn_softmax = nn.Softmax(dim=-1)
        self.add_module("atn_softmax", atn_softmax)

        # DCT stuff
        self.max_n = cnf["attention"]["dct"].get("maxN", None)
        self.max_m = cnf["attention"]["dct"].get("maxM", None)

        # Initialize class variable
        if DCTAttention.Q_dct is None and self.max_n is not None:
            DCTAttention.Q_dct = create_dct(n=self.max_n, m=self.max_m).to(cnf["model"]["device"])

    def forward(self, Q, K, V, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.transpose(2,3)
        self.device = V.device
        Q = Q / sqrt(sqrt(self.head_dim))
        K = K / sqrt(sqrt(self.head_dim)) * mask

        # Dct: (BS, NH, SL, E) -> (BS,NH, sl_, E)
        pad = 0
        if self.max_n is not None:
            Q = torch.matmul(self.Q_dct, Q)
            K = torch.matmul(self.Q_dct, K)
            V = torch.matmul(self.Q_dct, V * mask)
        else:
            pad = max(0, Q.shape[-2] - self.max_m)
            Q = dct(Q, dim=-2)[..., : self.max_m, :]
            K = dct(K, dim=-2)[..., : self.max_m, :]
            V = dct(V * mask, dim=-2)[..., : self.max_m, :]

        # (BS, NH, SL, E) * (BS,NH,E,sl_)
        energy = torch.matmul(Q, torch.transpose(K, -2, -1))
        attention = self.atn_softmax(energy)

        # E*V -> (bs. h, q, k) * (bs, h, v, head_dim) -> (bs, h, q, head_dim)
        # (note: k=v always!)
        if self.max_n is not None:
            x = torch.matmul(torch.matmul(self.Q_dct.t(), attention), V)
        else:
            x = torch.matmul(attention, V)
            x = torch.nn.ConstantPad1d((0, pad), 0)(x.transpose(-1, -2)).transpose(-1, -2)
            x = idct(x, dim=-2)

        return x, attention
