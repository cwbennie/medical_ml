import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models
from datetime import datetime
import cv2
import time
import utils
from models import ChexNet
from datasets import CheXpertData
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def setup(rank, world_size):
    # set up to use NCCL backend
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def model_setup(dataset: Dataset,
                batch_sz: int, rank, world_size):
    print(rank)
    model = ChexNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # assume dataset is evenly divisible by world_size
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                 rank=rank)

    dist_dataloader = DataLoader(dataset, batch_size=batch_sz,
                                 sampler=sampler)

    return ddp_model, dist_dataloader


def ave_auc(probs, ys):
    aucs = [roc_auc_score(ys[:, i], probs[:, i]) for i in range(probs.shape[1])]
    return np.mean(aucs), aucs


def cuda2cpu_classification(y: torch.Tensor): return y.long().cpu().numpy()


def cuda2cpu_regression(y: torch.Tensor): return y.cpu().numpy()


def validate_loop(model: nn.Module, valid_dl: DataLoader, task: str):
    if task == 'binary' or task == 'multilabel':
        cuda2cpu = cuda2cpu_classification
        loss_fun = F.binary_cross_entropy_with_logits
    elif task == 'regression':
        cuda2cpu = cuda2cpu_regression
        loss_fun = F.l1_loss

    model.eval()
    total = 0
    sum_loss = 0
    ys = []
    preds = []

    for x, y in valid_dl:
        out = model(x).squeeze()
        loss = loss_fun(out.squeeze(), y)

        batch = y.shape[0]
        sum_loss += batch * (loss.item())
        total += batch

        preds.append(out.squeeze().detach().cpu().numpy())
        ys.append(cuda2cpu(y))
    return sum_loss/total, preds, ys


def validate_multilabel(model, valid_dl):
    loss, preds, ys = validate_loop(model, valid_dl, 'multilabel')

    preds = np.vstack(preds)
    ys = np.vstack(ys)

    mean_auc, aucs = ave_auc(preds, ys)

    return loss, mean_auc, aucs


def train_chex_model_dist(rank, optimizer: torch.optim.Adam,
                          train_data: CheXpertData, valid_dl: DataLoader,
                          world_size, epochs: int = 25, track_loss=False,
                          lr_scheduler=None,
                          criterion: torch.nn.Module =
                          nn.BCEWithLogitsLoss(),
                          **kwargs):
    setup(rank, world_size)
    model, train_dl = model_setup(dataset=train_data, batch_sz=16,
                                  rank=rank, world_size=world_size)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer(parameters, lr=0.0001, betas=(0.9, 0.999))
    if lr_scheduler is not None:
        scheduler = utils.get_scheduler(name=lr_scheduler, optimizer=optimizer,
                                        epochs=epochs, max_lr=0.001,
                                        steps_per_epoch=len(train_dl),
                                        **kwargs)
    epoch_losses = list()
    auc_scores = list()
    val_losses = list()
    # criterion = criterion.to(device)

    # train the model for the given number of epochs
    for ep in range(epochs):
        train_dl.sampler.set_epoch(ep)
        losses = list()
        model.train()
        for i, (img, y) in enumerate(train_dl):
            img = img.to(rank).float()
            y = torch.stack(y, dim=1).to(rank).long()
            y = F.one_hot(y, num_classes=3).float()
            out = model(img)
            print(f'y device: {y.device}\noutput device: {out.device}')
            loss = criterion(out.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 1000 == 0 and rank == 0:
                print(f'Finished {i+1} iterations')
            break
        # epoch_losses.append(np.mean(losses))
        # print(f'Epoch finished: Loss -> {np.mean(losses)}')
        # if valid_dl and rank == 0:
        #     val_loss, epoch_auc, _ = validate_multilabel(model, valid_dl)
        #     val_losses.append(val_loss)
        #     auc_scores.append(epoch_auc)
        #     print(f'''Epoch Validation: Loss -> {val_loss}\n
        #           AUC Score -> {epoch_auc}\n''')
        # if lr_scheduler is not None:
        #     scheduler.step()

    cleanup()

    if track_loss and rank == 0:
        return epoch_losses, val_losses


### TODO
### able to parallelize, need to figure out gather function
### to track and update losses

def main():
    chex_dir = Path('/home/cwbennie/data/chexpert_xrays/CheXpert_v1.0_train')
    train_csv = '/home/cwbennie/data/chexpert_xrays/chexpert_csv_files/train.csv'
    chex_data = CheXpertData(chex_dir, train_csv, transform=True)    
    # train_dl = DataLoader(chex_data, batch_size=40, shuffle=True)

    valid_dir = Path(
        '/home/cwbennie/data/chexpert_xrays/CheXpert_v1.0_validation')
    val_csv = '/home/cwbennie/data/chexpert_xrays/chexpert_csv_files/valid.csv'
    val_data = CheXpertData(valid_dir, val_csv, transform=False,
                            data_type='valid')
    valid_dl = DataLoader(val_data, batch_size=8, shuffle=False)

    world_size = 2

    # model = model.to(device)
    mp.spawn(
        train_chex_model_dist,
        args=(torch.optim.Adam, chex_data, valid_dl,
              world_size, 25, False, 'OneCycle',
              nn.CrossEntropyLoss()),
        nprocs=world_size,
        join=True
    )
    # train_chex_model_dist(model, train_dl=train_dl, optimizer=torch.optim.Adam,
    #                       valid_dl=valid_dl, epochs=10,
    #                       lr_scheduler='OneCycle', max_lr=0.001,
    #                       steps_per_epoch=len(train_dl))


if __name__ == '__main__':
    main()
