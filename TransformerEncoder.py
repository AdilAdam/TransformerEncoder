#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn

from embedding import Posionalencoding, Embedding
from attention import Multiheadattention
from attention import PositionwiseFeedForward



class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_heads: int = 4,
        d_model: int = 256,
        hidden_size: int = 512,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.attn = Multiheadattention(d_model, n_heads, drop_rate)
        self.ffw = PositionwiseFeedForward(d_model, hidden_size, drop_rate)

    def forward(self, x):
        out, _ = self.attn(q=x, k=x, v=x, mask=None)
        out, xs = self.ffw(out)
        return out, xs


class Encoder(nn.Module):
    def __init__(
        self,
        num_layer: int = 4,
        vocab_size: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        hidden_size: int = 256,
        drop_rate: float = 0.5,
    ):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model)
        self.pos = Posionalencoding(d_model, drop_rate=drop_rate)
        self.layer = nn.ModuleList(
            EncoderBlock(n_heads, d_model, hidden_size, drop_rate=drop_rate)
            for _ in range(num_layer)
        )

    def forward(self, x):
        x = self.emb(x)
        x = self.pos(x)
        for layer in self.layer:
            out, xs = layer(x)
        return out, xs


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layer: int = 8,
        vocab_size: int = 50,
        n_heads: int = 8,
        d_model: int = 512,
        hidden_size: int = 1024,
        drop_rate: float = 0.5,
        num_class: int = 4,
    ):
        super(TransformerEncoder, self).__init__()
        self.encoder = Encoder(
            num_layer,
            vocab_size,
            d_model,
            n_heads,
            hidden_size,
            drop_rate,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifer = nn.Linear(d_model, num_class)
        self.activ = nn.Sigmoid()

    def forward(self, x):
        ffw_x, xs = self.encoder(x)
        x = ffw_x + xs
        x = self.layer_norm(x)
        x = self.classifer(x)
        return self.activ(x)