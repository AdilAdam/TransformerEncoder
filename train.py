import argparse
import yaml
from dataset import Dataset
from torch.utils.data import DataLoader
import copy
import torch
import torch.nn as nn
from models.vadmodel import init_model
from executor import Executor
from models.TransformerEncoder import TransformerEncoder


# def get_args():
#     parser = argparse.ArgumentParser("train")
#     parser.add_argument("--conf",type=str, default="/home/junlin/myproject/MyMOdel/config/config.yaml")
#     # parser.add_argument()
#     # parser.add_argument()

#     args= parser.parse_args()
#     return args

def main():
    # args = get_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    conf = "/home/junlin/myproject/MyMOdel/config/config.yaml"
    conf = yaml.load(open(conf, 'r'), Loader=yaml.FullLoader)
    model = init_model(conf)
    model = TransformerEncoder(conf)
    model.to(device)
    model_size = sum(p.numel() for p in model.parameters()) / (1024 * 1024)

    data = "./train_data.list"
    data_cv = "./dev_data.list"


    lossfn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', label_smoothing=0.5)
    pred_logmax = nn.LogSoftmax(dim=1)
    lr = 1e-3
    optermizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    train_data_conf = conf["dataset_conf"]
    cv_conf = copy.deepcopy(train_data_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['shuffle'] = False
    train_data = Dataset(data, conf=train_data_conf)

    train_data_loader = DataLoader(train_data,
                                    batch_size=None,
                                    pin_memory=False,
                                    num_workers=2,
                                    prefetch_factor=1)
    
    cv_data = Dataset(data_cv, conf=cv_conf)
    cv_data_loader = DataLoader(cv_data,
                                    batch_size=None,
                                    pin_memory=False,
                                    num_workers=2,
                                    prefetch_factor=1)
    print(model)
    print(model_size)
    for epoch in range(5):
        for batch_idx, bath in enumerate(train_data_loader):
            sorted_keys, padded_feats, padded_labels, feats_lengths, label_lengths = bath
           
            loss_train = []
            loss_cv = []
            model.train()
            padded_feats = padded_feats.to(device)
            padded_labels = padded_labels.to(device)
            feats_lengths = feats_lengths.to(device)

            pred_ = model(padded_feats, feats_lengths)
            pred = pred_.transpose(2, 1)
            loss = lossfn(pred, padded_labels.long())
            optermizer.zero_grad()
            loss.backward()
            optermizer.step()
            if batch_idx % 100 == 0:
                print("train_loss: {:.3f}".format(loss.detach().float().tolist()))
                loss_train.append(loss.item())


        for batch_idx, bath in enumerate(cv_data_loader):
            cv_sorted_keys, cv_padded_feats, cv_padded_labels, cv_feats_lengths, cv_label_lengths = bath
            
            model.eval()
            with torch.no_grad():
                cv_padded_feats = cv_padded_feats.to(device)
                cv_padded_labels = cv_padded_labels.to(device)
                cv_feats_lengths = cv_feats_lengths.to(device)
                cv_pred = model(cv_padded_feats, cv_feats_lengths)
                cv_pred = cv_pred.transpose(2, 1)
                cv_loss = lossfn(cv_pred, cv_padded_labels.long())
                if batch_idx%100==0:
                    print("cv_loss: {:.3f}".format(cv_loss.detach().float().tolist()))
                    loss_cv.append(cv_loss.item())


if __name__=="__main__":

    main()