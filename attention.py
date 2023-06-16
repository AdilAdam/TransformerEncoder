#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, tempereture: float, drop_rate: float = 0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = tempereture
        self.drop_out = nn.Dropout(p=drop_rate)

    def forward(self, q, k, v, mask=None) -> torch.Tensor:
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.drop_out(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class Multiheadattention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, drop_rate: float = 0.0):
        super(Multiheadattention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = self.d_v = d_model // n_heads
        self.h = n_heads
        self.q_w = nn.Linear(d_model, self.h * self.d_k)
        self.k_w = nn.Linear(d_model, self.h * self.d_k)
        self.v_w = nn.Linear(d_model, self.h * self.d_v)
        self.o_w = nn.Linear(self.d_v * self.h, d_model)
        self.self_attn = ScaledDotProductAttention(
            tempereture=self.d_k**0.5, drop_rate=0.1
        )
        self.dropout = nn.Dropout(p=drop_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # q,k,v : [batch_size, sequence_len, d_model]
        residual = q
        n_batch, seq_len = q.size(0), q.size(1)
        q_w = self.q_w(q).view(n_batch, seq_len, self.h, self.d_k)
        k_w = self.k_w(k).view(n_batch, seq_len, self.h, self.d_k)
        v_w = self.v_w(v).view(n_batch, seq_len, self.h, self.d_k)
        q, k, v = q_w.transpose(1, 2), k_w.transpose(1, 2), v_w.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        out, attn = self.self_attn(q=q, k=k, v=v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(n_batch, seq_len, -1)
        out = self.dropout(self.o_w(out))
        out += residual
        out = self.layer_norm(out)

        return out, attn


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.
    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        drop_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(drop_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs)))), xs
