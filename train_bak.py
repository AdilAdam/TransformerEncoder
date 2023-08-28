import os
import argparse
import json

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
import yaml
from matplotlib.pylab import plt
from numpy import arange
from torch.autograd import Variable
from utlis.conf import Conf

from tensorboardX import SummaryWriter
from models.scheduler import WarmupLR
from models.TransformerEncoder import TransformerEncoder


def get_args():
    parser = argparse.ArgumentParser(description="A transformer encoder")
    parser.add_argument("--conf", type=str, 
                        default="/home/junlin/myproject/MyMOdel/config/config.yaml",
                        help="configure file")

    args = parser.parse_args()

    return args

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    np_mask = Variable(torch.from_numpy(np_mask == 1))
    return np_mask

def create_masks(src, trg, device):
    src_mask = (src != 1).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != 1).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size).to(device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


def main():

    args = get_args()
    # device = (torch.device("cuda") 
    #           if torch.cuda.is_available() else torch.device("cpu"))
    device = torch.device("cpu")
    conf = yaml.load(open(args.conf), Loader=yaml.FullLoader )
    conf["model"]["device"] = device
    model = TransformerEncoder(conf)
    model.to(device)

    data = json.load(open(conf["dataset"]["train"]))
    train_data = data["data"]["train"]

    exp_dir = os.path.join(Path(__file__).parent, "exp")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    writer = SummaryWriter(exp_dir)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **conf["optim_conf"])
    scheduler = WarmupLR(optimizer, **conf["scheduler_conf"])


    
# ###################data processing######################

    train_data = data["data"]["train"]
    label = data["data"]["label"]
    cv_data = data["data"]["valid"]
    cv_label = data["data"]["v_label"]
    data = torch.tensor(train_data).to(device)
    label = torch.tensor(label).to(device)
    cv_data = torch.tensor(cv_data).to(device)
    cv_label = torch.tensor(cv_label).to(device)
    print(cv_data.shape, cv_label.shape)
    loss_train = []
    loss_cv = []

    data = torch.cat(
        (
            data,
            data,
            data,
            data
        )
    )
    label = torch.cat(
        (
            label,
            label,
            label,
            label
        )
    )
    cv_data = torch.cat(
        (
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
        )
    )

   

    src_mask, trg_mask = create_masks(data, label, device)
    cv_mask, cv_trg = create_masks(cv_data, cv_label, device)
    print(src_mask.squeeze(1).shape, "$$$$$$")
# # ###########################################

    # for epoch in range(2):
    #     tr_correct = 0
    #     tr_total_sample = 0
    #     for i in range(1000):
           
    #         model.train()
    #         pred = model(data, mask=src_mask.squeeze(1))
    #         pred = pred.transpose(2, 1)
    #         pred_ = pred.data.max(1, keepdim=True)[1].squeeze()
    #         # print(pred_.shape, label.shape)
    #         tr_total_sample += label.size(0) * 60
    #         tr_correct += pred_.eq(label.data.view_as(pred_)).sum().item()

    #         train_acc = (tr_correct / tr_total_sample) * 100
        
    #         loss = loss_fn(pred, label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         if i % 100 == 0:
    #             print("train_loss: {:.3f}".format(loss.detach().float().tolist()))
    #             loss_train.append(loss.item())

    #         model.eval()
    #         with torch.no_grad():
    #             cv_pred = model(cv_data, mask=cv_mask.squeeze(1))
    #             cv_pred = cv_pred.transpose(2, 1)
    #             cv_loss = loss_fn(cv_pred, cv_label)
    #             if i % 100 == 0:
    #                 print("cv_loss: {:.3f}".format(cv_loss.detach().float().tolist()))
    #                 loss_cv.append(cv_loss.item())
    #     state_dict = model.state_dict()
    #     print(train_acc)
    #     torch.save(state_dict, "results/checkpoint_dct" + str(epoch) + ".pt")



    # plt.plot(np.array(loss_train), label="Train Loss", marker="o")
    # plt.plot(np.array(loss_cv), label="valid Loss", marker="x")
    # plt.title("Training and Validation Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.xticks(arange(0, 20, 2))
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.savefig("results/loss_1.png")
    # plt.show()


if __name__ == "__main__":
    main()


