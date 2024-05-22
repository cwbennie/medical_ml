import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import albumentations as A
from pathlib import Path
import matplotlib.pyplot as plt
import os
import re
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# Define Utilities
def read_image(path):
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r, c, *_ = im.shape
    M = cv2.getRotationMatrix2D((c/2, r/2), deg, 1)
    return cv2.warpAffine(im, M, (c, r), borderMode=mode,
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)


def resize_image(path: Path, size: int):
    image = cv2.imread(str(path))
    return cv2.resize(image, (size, size))


def resize_all_images(dir_path: Path, size: int):
    new_path = Path(f'{str(dir_path)}_{str(size)}/')
    for dirc in dir_path.iterdir():
        dir_name = str(dirc).split('/')[-1]
        if dir_name != '.DS_Store':
            dir_path = new_path/dir_name
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f'Directory {dir_path} created successfully')
            for image in dirc.iterdir():
                new_img = resize_image(image, size)
                path = dir_path/image.name
                cv2.imwrite(str(path), new_img)


def rotate_img(path: Path, rot: int):
    image = cv2.imread(str(path))
    return np.rot90(image, rot)


def get_mura_category(filename: str):
    xray_pattern = r'XR_([A-Z]+)'
    match = re.search(pattern=xray_pattern, string=filename)
    return match.group(1)


def rotate_all_images(dir_path: Path, rot: int):
    new_path = Path(f'{str(dir_path)}_rotations/{str(int(rot*90))}_deg/')
    for dirc in dir_path.iterdir():
        dir_name = str(dirc).split('/')[-1]
        if dir_name != '.DS_Store':
            dir_path = new_path/dir_name
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f'Directory {dir_path} created successfully')
            for image in dirc.iterdir():
                new_img = rotate_img(image, rot)
                path = dir_path/image.name
                cv2.imwrite(str(path), new_img)


# function to create lists of images and labels from directories of images
# this will be used in the Datasets
def get_images_labels(path: Path, classes: dict):
    images = list()
    labels = list()
    for group, label in classes.items():
        class_images = list()
        for dirpath, dirnames, files in os.walk(Path(path/group)):
            for file in files:
                if file.endswith('png'):
                    im_file = os.path.join(dirpath, file)
                    img = cv2.imread(str(im_file)).astype(np.float32)
                    class_images.append(img)
        class_labels = label * np.ones(len(class_images), dtype=int)
        images += class_images
        if len(labels) > 0:
            labels = np.concatenate((labels, class_labels), axis=0)
        else:
            labels = class_labels
    return images, labels


# function to create lists of images and labels from directories of images
def get_rotation_labels(path: Path, classes: dict):
    images = list()
    labels = list()
    for group, label in classes.items():
        class_images = list()
        for dirpath, dirnames, files in os.walk(Path(path/group)):
            for file in files:
                if file.endswith('png'):
                    class_images.append(os.path.join(dirpath, file))
        class_labels = label * np.ones(len(class_images), dtype=int)
        images += class_images
        if len(labels) > 0:
            labels = np.concatenate((labels, class_labels), axis=0)
        else:
            labels = class_labels
    return images, labels


# function to create lists of images and labels from directories of images
def get_images(path: Path):
    images = list()
    for dirpath, dirnames, files in os.walk(Path(path)):
        for file in files:
            if file.endswith('png'):
                im_file = os.path.join(dirpath, file)
                img = cv2.imread(str(im_file)).astype(np.float32)
                images.append(img)
    return images


def get_rotations(images):
    new_imgs = list()
    new_labels = list()
    for rot in [0, 1, 2, 3]:
        for image in images:
            rot_im = np.rot90(image, rot)
            new_imgs.append(rot_im)
            new_labels.append(rot)
    return new_imgs, np.array(new_labels)


def flatten(x: np.array):
    return x.reshape((x.shape[0]), -1)[0]


def show_img(path):
    im = cv2.imread(str(path))
    plt.imshow(im)


def classSampling(X, y, samplesPerClass, numberOfClasses):
    X_ret = list()
    y_ret = list()

    for classIdx in range(numberOfClasses):
        indices = np.where(np.array(y) == classIdx)[0]

        doResample = len(indices) < samplesPerClass

        chosenIndices = np.random.choice(indices, samplesPerClass,
                                         replace=doResample)

        for ci in chosenIndices:
            X_ret.append(X[ci])
            y_ret.append(y[ci])

    return X_ret, y_ret


def save_model(model: nn.Module, model_name: str):
    # function to save model weights
    path = '/home/cwbennie/medical_ml/models/'+model_name
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def get_scheduler(name, optimizer, **kwargs):
    if name == 'Step Decay':
        # StepLR requires step_size: argument can be passed with kwargs
        scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif name == 'Exponential Decay':
        # ExponentialLR requires gamma: argument passed with kwargs
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        # OneCycle requires max_lr, steps_per_epoch, epochs: pass with kwargs
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    return scheduler


# evaluate the performance of the model on validation images
def model_eval(model, valid_dl, test=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    predictions = list()
    y_actuals = list()
    probs = list()
    losses = list()
    model.eval()
    for x_val, y_val in valid_dl:
        out = model(x_val.to(device))
        loss = F.cross_entropy(out, y_val.to(device))
        losses.append(loss.item())
        _, preds = torch.max(out, 1)
        predictions.extend(preds.cpu().detach().numpy())
        probs.extend(out.cpu().detach().numpy())
        y_actuals.extend(y_val.numpy())
    if not test:
        return np.mean(losses)
    return predictions, y_actuals, probs


# function to fine-tune ResNet on the sonar images
def train_model(model: nn.Module, optimizer: torch.optim.Adam,
                train_dl, valid_dl, epochs: int = 25, track_loss=False,
                lr_scheduler=None, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer(parameters, lr=0.001, betas=(0.5, 0.999), eps=1)
    if lr_scheduler is not None:
        scheduler = get_scheduler(name=lr_scheduler, optimizer=optimizer,
                                  **kwargs)
    epoch_losses = list()
    val_losses = list()

    # train the model for the given number of epochs
    for i in tqdm(range(epochs), desc='Training', leave=True):
        losses = list()
        for x, y in train_dl:
            model.train()
            x = x.to(device).float()
            y = y.to(device).long()
            out = model(x)
            # _, preds = torch.max(out, 1)
            loss = F.cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(np.mean(losses))
        val_loss = model_eval(model, valid_dl, test=False)
        val_losses.append(val_loss)
        if lr_scheduler is not None:
            scheduler.step()
    if track_loss:
        return epoch_losses, val_losses


def update_image_dict(im_file: Path, image_dict: dict):
    img = cv2.imread(str(im_file))
    if img is not None:
        # img = img.astype(np.float32)
        if image_dict.get('images'):
            image_dict['images'].append(im_file)
        else:
            image_dict['images'] = [im_file]


def get_mura_images(path: Path):
    images = list()
    labels = list()
    categories = list()
    for dirpath, _, files in os.walk(Path(path)):
        for file in files:
            if file.endswith('.png'):
                im_file = os.path.join(dirpath, file)
                img = cv2.imread(str(im_file))
                if img is not None:
                    images.append(im_file)
                    label = 1 if 'positive' in str(dirpath) else 0
                    labels.append(label)
                    cat = get_mura_category(str(dirpath))
                    categories.append(cat)
    return images, labels, categories


def transform_image(image, transform_pipe: A.Compose):
    img = cv2.imread(str(image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform_pipe(image=img)
    return augmented['image']


def get_category_counts(labels: list, categories: list):
    count_dict = dict()
    neg_dict = dict()
    for i in range(len(labels)):
        if labels[i] == 0:
            neg_dict[categories[i]] = neg_dict.get(categories[i], 0) + 1
        else:
            count_dict[categories[i]] = count_dict.get(categories[i], 0) + 1
    pos_weights = {key: count_dict[key] / (count_dict[key] + neg_dict[key])
                   for key in count_dict.keys()}
    neg_weights = {key: neg_dict[key] / (count_dict[key] + neg_dict[key])
                   for key in count_dict.keys()}
    return pos_weights, neg_weights


def compute_auc(probs, y_vals):
    probs = np.vstack(probs)
    y_vals = np.vstack(y_vals)
    return roc_auc_score(y_true=y_vals, y_score=probs)


class MURALoss(torch.nn.Module):
    def __init__(self):
        super(MURALoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                weights: tuple):
        loss_pos_wts = torch.Tensor(weights[0]).to(self.device).float()
        loss_neg_wts = torch.Tensor(weights[1]).to(self.device).float()
        loss = -(loss_pos_wts * targets * predictions.log() +
                 loss_neg_wts * (1 - targets) * (1 - predictions).log())
        return loss.mean()


def mura_model_eval(model: nn.Module, valid_dl: DataLoader,
                    criterion: MURALoss, test=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    predictions = list()
    y_actuals = list()
    probs = list()
    losses = list()
    model.eval()
    for x_val, y_val, weights in valid_dl:
        out = model(x_val.to(device))
        y_val = y_val.to(device).float()
        loss = criterion(out.squeeze(), y_val, weights)
        losses.append(loss.item())
        preds = out.squeeze()
        predictions.extend(preds.cpu().detach().numpy())
        probs.extend(out.cpu().detach().numpy())
        y_actuals.extend(y_val.cpu().numpy())
    auc_score = compute_auc(probs=probs, y_vals=y_actuals)
    if not test:
        return np.mean(losses)
    return np.mean(losses), auc_score


# function to fine-tune ResNet on the sonar images
def train_mura_model(model: nn.Module, optimizer: torch.optim.Adam,
                     train_dl, valid_dl, epochs: int = 25, track_loss=False,
                     lr_scheduler=None, criterion: torch.nn.Module = MURALoss,
                     **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer(parameters, lr=0.0001, betas=(0.9, 0.999))
    if lr_scheduler is not None:
        scheduler = get_scheduler(name=lr_scheduler, optimizer=optimizer,
                                  epochs=epochs, **kwargs)
    epoch_losses = list()
    auc_scores = list()
    val_losses = list()
    criterion = criterion.to(device)

    # train the model for the given number of epochs
    for i in tqdm(range(epochs), desc='Training', leave=True):
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
        print(f'Epoch finished: Loss -> {np.mean(losses)}')
        if valid_dl:
            val_loss, epoch_auc = mura_model_eval(model, valid_dl,
                                                  criterion=criterion,
                                                  test=True)
            val_losses.append(val_loss)
            auc_scores.append(epoch_auc)
            print(f'''Epoch Validation: Loss -> {val_loss}\n
                  AUC Score -> {epoch_auc}\n''')
        if lr_scheduler is not None:
            scheduler.step()
    if track_loss:
        return epoch_losses, val_losses


def cosine_segment(start_lr, end_lr, iterations):
    i = np.arange(iterations)
    c_i = 1 + np.cos(i*np.pi/iterations)
    return end_lr + (start_lr - end_lr)/2 * c_i


def get_cosine_triangular_lr(max_lr, iterations):
    min_start, min_end = max_lr/25, max_lr/(25*1e4)
    iter1 = int(0.3*iterations)
    iter2 = iterations - iter1
    segs = [cosine_segment(min_start, max_lr, iter1),
            cosine_segment(max_lr, min_end, iter2)]
    return np.concatenate(segs)


def diff_lr(lr, alpha=1/3):
    return [lr*alpha**i for i in range(2, -1, -1)]


def create_optimizer(model, lr_0):
    param_groups = [list(model.groups[i].parameters()) for i in range(3)]
    params = [{'params': p, 'lr': lr} for p, lr
              in zip(param_groups, diff_lr(lr_0))]
    return optim.Adam(params)


def update_optimizer(optimizer, group_lrs):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = group_lrs[i]
