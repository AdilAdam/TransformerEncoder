import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        return x


class Posionalencoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        reverse: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super(Posionalencoding, self).__init__()
        self.xscale = math.sqrt(d_model)
        self.pe = torch.zeros(max_len, d_model)

        self.drop_out = nn.Dropout(p=drop_rate)
        if reverse:
            position = torch.arange(
                max_len - 1, -1, 1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

        # self.register_buffer("pe", self.pe, persistent=False)

    def forward(self, x: torch.Tensor):
        self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.drop_out(x)

