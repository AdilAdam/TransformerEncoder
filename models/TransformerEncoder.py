#!/usr/bin/env python

import json
import yaml
import numpy as np
import torch
from typing import Tuple
import torch.nn as nn
import sys
sys.path.append(".")
from models.attentions.multiheadattention import Multiheadattention
from models.attentions.PositionwiseFeedForward import PositionwiseFeedForward
from models.embedding import Embedding, PositionalEncoding, LinearNoSubsampling
from models.scheduler import WarmupLR
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from models.embedding import make_pad_mask
# from models.vadmodel import add_sos_eos, reverse_pad_list, th_accuracy




class EncoderBlock(nn.Module):
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


class Encoder(nn.Module):
    def __init__(
        self,
        conf,
        num_layer: int = 4,
        input_size: int= 80,
        d_model: int = 256,
        n_heads: int = 4,
        hidden_size: int = 512,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.pos = LinearNoSubsampling(input_size,
            d_model,
            drop_rate,
            (PositionalEncoding(d_model, dropout_rate=drop_rate)))
        self.layenorm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.layer = nn.ModuleList(
            EncoderBlock(conf, n_heads, d_model, hidden_size, drop_rate=drop_rate) for _ in range(num_layer)
        )

    def forward(self, xs, masks):

        xs, pos_emb, masks = self.pos(xs, masks)
        for layer in self.layer:
            out, xs = layer(xs, mask=masks)
        return out, xs, masks


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        conf
    ):
        super(TransformerEncoder, self).__init__()
       
        num_layer = conf["model"].get("num_layer", 8)
        input_size = conf["dataset_conf"]["feature_extraction_conf"]["num_mel_bins"]
        n_heads = conf["model"].get("n_heads", 8)
        d_model = conf["model"].get("d_model", 512)
        hidden_size = conf["model"].get("hidden_size", 1024)
        drop_rate = conf["model"].get("drop_rate", 0.1)
        num_class = conf["model"].get("num_class", 2)


        self.encoder = Encoder(
            conf=conf,
            num_layer=num_layer,
            input_size=input_size,
            d_model=d_model,
            n_heads=n_heads,
            hidden_size=hidden_size,
            drop_rate=drop_rate,
        )
        
        self.lstm = nn.LSTM(input_size =d_model, hidden_size= d_model, num_layers=2, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifer = nn.Linear(d_model, num_class)
        # self.activ = nn.Sigmoid()
        self.activ = nn.Identity()

    def forward(self, 
                xs: torch.Tensor,
                xs_lens: torch.Tensor,
                ):
        """
        xs : input features (B, T, D)
        xs_lens: feats length
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        ffw_x, xs, masks = self.encoder(xs, masks)
        x = ffw_x + xs
        x = self.lstm(x)
        x = self.layer_norm(x[0])
        x = self.classifer(x)
        return self.activ(x)
    

   


if __name__ == "__main__":

    
    conf = "/home/junlin/myproject/MyMOdel/config/conf.json"
    model = TransformerEncoder(conf)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    print(model)
    model_size = sum(p.numel() for p in model.parameters()) / (1024 * 1024)
    data = "/home/junlin/myproject/MyMOdel/data/data.json"
    f = open(data)
    data = json.load(f)
    train_data = data["data"]["train"]
    lossfn = nn.CrossEntropyLoss()
    pred_logmax = nn.LogSoftmax(dim=1)
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupLR(optimizer, warmup_steps=300)
    train_data = data["data"]["train"]
    label = data["data"]["label"]
    cv_data = data["data"]["valid"]
    cv_label = data["data"]["v_label"]
    data = torch.tensor(train_data).to(device)
    label = torch.tensor(label).to(device)
    cv_data = torch.tensor(cv_data).to(device)
    cv_label = torch.tensor(cv_label).to(device)
    loss_train = []
    loss_cv = []
    model.to(device)

    data = torch.cat(
        (
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            data,
        )
    )
    label = torch.cat(
        (
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
            label,
        )
    )
    cv_data = torch.cat(
        (
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
            cv_data,
        )
    )
    cv_label = torch.cat(
        (
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
            cv_label,
        )
    )

    def nopeak_mask(size):
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
        np_mask = Variable(torch.from_numpy(np_mask == 1))
        return np_mask

    def create_masks(src, trg):
        src_mask = (src != 1).unsqueeze(-2)
        if trg is not None:
            trg_mask = (trg != 1).unsqueeze(-2)
            size = trg.size(1)  # get seq_len for matrix
            np_mask = nopeak_mask(size).to(device)
            trg_mask = trg_mask & np_mask

        else:
            trg_mask = None
        return src_mask, trg_mask

    src_mask, trg_mask = create_masks(data, label)
    cv_mask, cv_trg = create_masks(cv_data, cv_label)
    for epoch in range(2):
        for i in range(1000):
            model.train()
            pred = model(data, mask=src_mask.squeeze(1))
            pred = pred.transpose(2, 1)
            loss = lossfn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                print("train_loss: {:.3f}".format(loss.detach().float().tolist()))
                loss_train.append(loss.item())

            model.eval()
            with torch.no_grad():
                cv_pred = model(cv_data, mask=cv_mask.squeeze(1))
                cv_pred = cv_pred.transpose(2, 1)
                cv_loss = lossfn(cv_pred, cv_label)
                if i % 100 == 0:
                    print("cv_loss: {:.3f}".format(cv_loss.detach().float().tolist()))
                    loss_cv.append(cv_loss.item())
        state_dict = model.state_dict()
        torch.save(state_dict, "results/checkpoint_dct" + str(epoch) + ".pt")

    import numpy as np
    from matplotlib.pylab import plt
    from numpy import arange

    plt.plot(np.array(loss_train), label="Train Loss", marker="o")
    plt.plot(np.array(loss_cv), label="valid Loss", marker="x")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(arange(0, 20, 2))
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("results/loss_1.png")
    plt.show()
