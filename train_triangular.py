import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
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
from models import MURATriangular, MURALoss
from datasets import MuraData


def train_triangular_policy(model, train_dl, valid_dl, criterion,
                            max_lr=0.04, epochs=5,
                            output_file: str =
                            '/home/cwbennie/MURA_logs/triangular_train.txt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx = 0
    iterations = epochs*len(train_dl)
    lrs = utils.get_cosine_triangular_lr(max_lr, iterations)
    optimizer = utils.create_optimizer(model, lrs[0])
    prev_val_auc = 0.0
    for j in range(epochs):
        model.train()
        losses = list()
        for i, (img, y, wts) in enumerate(train_dl):
            lr = lrs[idx]
            utils.update_optimizer(optimizer, [lr/9, lr/3, lr])
            img = img.to(device).float()
            y = y.to(device).float()
            out = model(img)
            loss = criterion(out.squeeze(), y, wts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            idx += 1
        train_loss = np.mean(losses)
        val_loss, val_auc = utils.mura_model_eval(model, valid_dl,
                                                  criterion)
        with open(output_file, 'a') as file:
            file.write(f'''Epoch {j+1}\n
              Train: train_loss {train_loss:.3f}\n
              Val: val_loss {val_loss:.3f} val_auc {val_auc:.3f} \n''')
        if val_auc > prev_val_auc:
            prev_val_auc = val_auc
            path = f"model_triangular_auc_1_{100*val_auc:.0f}.pth"
            utils.save_model(model, path)
            with open(output_file, 'a') as file:
                file.write(f'''Model: {path}''')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pth = Path('/home/cwbennie/data/MURA-v1.1/train')
    mura_train = MuraData(pth, transform=True)
    train_dl = DataLoader(mura_train, batch_size=8, shuffle=True)

    val_path = Path('/home/cwbennie/data/MURA-v1.1/valid')
    mura_test = MuraData(val_path, transform=False)
    valid_dl = DataLoader(mura_test, batch_size=8, shuffle=False)

    model = MURATriangular()
    model = model.to(device)
    loss = MURALoss()
    train_triangular_policy(model=model, train_dl=train_dl,
                            valid_dl=valid_dl, criterion=loss,
                            max_lr=0.04, epochs=10)


if __name__ == '__main__':
    main()
