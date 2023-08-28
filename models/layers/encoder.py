#!/usr/bin/env python

import torch.nn as nn
from attentions.multiheadattention import Multiheadattention
from attentions.PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        conf,
        n_heads: int = 4,
        d_model: int = 256,
        hidden_size: int = 512,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.attn = Multiheadattention(conf, d_model, n_heads, drop_rate)
        self.ffw = PositionwiseFeedForward(d_model, hidden_size, drop_rate)

    def forward(self, x, mask=None):
        out, _ = self.attn(Q=x, K=x, V=x, mask=mask)
        out, xs = self.ffw(out)
        return out, xs
