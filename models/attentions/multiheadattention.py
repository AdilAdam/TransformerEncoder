#!/usr/bin/env python


import torch
import torch.nn as nn
from models.attentions.attention import DCTAttention, ScaledDotProductAttention


class Multiheadattention(nn.Module):
    def __init__(self, conf: str, d_model: int, n_heads: int, drop_rate: float = 0.0):
        super(Multiheadattention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = self.d_v = d_model // n_heads
        self.h = n_heads
        self.q_w = nn.Linear(d_model, self.h * self.d_k)
        self.k_w = nn.Linear(d_model, self.h * self.d_k)
        self.v_w = nn.Linear(d_model, self.h * self.d_v)
        self.o_w = nn.Linear(self.d_v * self.h, d_model)

        attention_type = conf["attention"].get("type")
        if attention_type == "ScaledDotProductAttention":
            self.self_attn = ScaledDotProductAttention(
                tempereture=self.d_k**0.5, drop_rate=0.1
            )
        elif attention_type == "DCTAttention":
            self.self_attn = DCTAttention(cnf=conf)

        elif attention_type == "Linerattention":
            self.self_attn = None # 
            
        self.dropout = nn.Dropout(p=drop_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # q,k,v : [batch_size, sequence_len, d_model]
        residual = Q
        n_batch, seq_len = Q.size(0), Q.size(1)
        q_w = self.q_w(Q).view(n_batch, seq_len, self.h, self.d_k)
        k_w = self.k_w(K).view(n_batch, seq_len, self.h, self.d_k)
        v_w = self.v_w(V).view(n_batch, seq_len, self.h, self.d_k)
        q, k, v = q_w.transpose(1, 2), k_w.transpose(1, 2), v_w.transpose(1, 2)
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        out, attn = self.self_attn(Q=q, K=k, V=v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(n_batch, seq_len, -1)
        out = self.dropout(self.o_w(out))
        out += residual
        out = self.layer_norm(out)

        return out, attn
