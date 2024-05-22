import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from datetime import datetime
import cv2
import utils
from models import MURALoss, MURANet
from datasets import MuraData


def train_mura_model(model: nn.Module, optimizer: torch.optim.Adam,
                     train_dl, valid_dl, epochs: int = 25, track_loss=False,
                     lr_scheduler=None, criterion: torch.nn.Module = MURALoss,
                     **kwargs):
    output_file = '/home/cwbennie/MURA_logs/onecycle_train.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer(parameters, lr=0.0001, betas=(0.9, 0.999))
    if lr_scheduler is not None:
        scheduler = utils.get_scheduler(name=lr_scheduler, optimizer=optimizer,
                                        epochs=epochs, **kwargs)
    epoch_losses = list()
    auc_scores = list()
    val_losses = list()
    criterion = criterion.to(device)
    prev_val_auc = 0.0

    # train the model for the given number of epochs
    for i in range(epochs):
        losses = list()
        model.train()
        for img, y, weights in train_dl:
            img = img.to(device).float()
            y = y.to(device).float()
            out = model(img)
            loss = criterion(out.squeeze(), y, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(np.mean(losses))
        if valid_dl:
            val_loss, epoch_auc = utils.mura_model_eval(model, valid_dl,
                                                        criterion=criterion,
                                                        test=True)
            val_losses.append(val_loss)
            auc_scores.append(epoch_auc)
            print(f'''Epoch Validation: Loss -> {val_loss}\n
                  AUC Score -> {epoch_auc}\n''')
        with open(output_file, 'a') as file:
            file.write(f'''Epoch {i+1}\n
              Train: train_loss {np.mean(losses):.3f}\n
              Val: val_loss {val_loss:.3f} val_auc {epoch_auc:.3f} \n''')
        if epoch_auc > prev_val_auc:
            prev_val_auc = epoch_auc
            path = f"model_onecycle_auc_1_{100*epoch_auc:.0f}.pth"
            utils.save_model(model, path)
            with open(output_file, 'a') as file:
                file.write(f'''Model: {path}''')
        if lr_scheduler is not None:
            scheduler.step()
    if track_loss:
        return epoch_losses, val_losses


def main():
    pth = Path('/home/cwbennie/data/MURA-v1.1/train')
    mura_train = MuraData(pth, transform=True)
    train_dl = DataLoader(mura_train, batch_size=8, shuffle=True)

    val_path = Path('/home/cwbennie/data/MURA-v1.1/valid')
    mura_test = MuraData(val_path, transform=False)
    valid_dl = DataLoader(mura_test, batch_size=8, shuffle=False)

    model = MURANet()
    loss = MURALoss()
    train_mura_model(model, train_dl=train_dl, optimizer=torch.optim.Adam,
                     valid_dl=valid_dl, epochs=10, criterion=loss,
                     lr_scheduler='OneCycle', max_lr=0.001,
                     steps_per_epoch=len(train_dl))


if __name__ == '__main__':
    main()
