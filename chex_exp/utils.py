import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import numpy as np
import albumentations as A
from pathlib import Path
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import cv2
import time
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, auc, roc_curve
from tqdm import tqdm
import wandb


# Define Utilities
def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """
    Rotates an image by a specified degree.

    Parameters:
    - im (numpy.ndarray): Input image.
    - deg (float): Degree to rotate the image.
    - mode (int): Border mode to use during rotation
                  (default is cv2.BORDER_REFLECT).
    - interpolation (int): Interpolation method to use
                           (default is cv2.INTER_AREA).

    Returns:
    - numpy.ndarray: Rotated image.
    """
    r, c, *_ = im.shape
    M = cv2.getRotationMatrix2D((c/2, r/2), deg, 1)
    return cv2.warpAffine(im, M, (c, r), borderMode=mode,
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)


def resize_image(path: Path, size: int):
    """
    Resizes an image to a specified size.

    Parameters:
    - path (Path): Path to the image file.
    - size (int): New size for both dimensions of the image.

    Returns:
    - numpy.ndarray: Resized image.
    """
    image = cv2.imread(str(path))
    return cv2.resize(image, (size, size))


def resize_all_images(dir_path: Path, size: int):
    """
    Resizes all images in a specified directory to a given size and saves
    them in a new directory.

    Parameters:
    - dir_path (Path): Directory containing images to resize.
    - size (int): New size for both dimensions of each image.

    Returns:
    - None: Images are saved in a new directory, named as the original
            directory followed by the specified size.
    """
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
    """
    Rotates an image by 90 degrees a specified number of times.

    Parameters:
    - path (Path): Path to the image file.
    - rot (int): Number of times to rotate the image by 90 degrees.

    Returns:
    - numpy.ndarray: Rotated image.
    """
    image = cv2.imread(str(path))
    return np.rot90(image, rot)


def get_mura_category(filename: str):
    """
    Extracts the MURA category from a filename using a regular expression.
    The categories will be used to compare performance across different
    image types.

    Parameters:
    - filename (str): Filename to extract the category from.

    Returns:
    - str: MURA category extracted from the filename.
    """
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
    path = '/home/cwbennie/MURA_models/'+model_name
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def get_scheduler(name, optimizer, **kwargs):
    if name == 'step_decay':
        # StepLR requires step_size: argument can be passed with kwargs
        scheduler = optim.lr_scheduler.StepLR(optimizer, kwargs['step_size'])
    elif name == 'exp_decay':
        # ExponentialLR requires gamma: argument passed with kwargs
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     kwargs['gamma'])
    else:
        # OneCycle requires max_lr, steps_per_epoch, epochs: pass with kwargs
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, kwargs['max_lr'],
                                                  kwargs['steps_per_epoch'],
                                                  kwargs['epochs'])
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


def train_model(model: nn.Module, optimizer: torch.optim.Adam,
                train_dl, valid_dl, epochs: int = 10, track_loss=False,
                lr_scheduler=None, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer(parameters, lr=0.0001, betas=(0.9, 0.999), eps=1)
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
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                weights: tuple):
        epsilon = 1e-6
        predictions = torch.clamp(predictions, min=epsilon, max=1-epsilon)
        loss_pos_wts = torch.Tensor(weights[0]).to(self.device).float()
        loss_neg_wts = torch.Tensor(weights[1]).to(self.device).float()
        loss = -(loss_pos_wts * targets * predictions.log() +
                 loss_neg_wts * (1 - targets) * (1 - predictions).log())
        return loss.mean()


def mura_model_eval(model: nn.Module, valid_dl: DataLoader,
                    criterion: MURALoss, test=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = list()
    y_actuals = list()
    probs = list()
    losses = list()
    model.eval()
    for x_val, y_val, weights in valid_dl:
        out = model(x_val.to(device).float())
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
    return np.mean(losses), auc_score, predictions, y_actuals


def log_train(output_file: str, elapsed: float, losses: list,
              val_loss: float, epoch_auc: float, epoch_num: int):
    with open(output_file, 'a') as file:
        file.write(f'''Epoch {epoch_num+1} - Time: {elapsed:.3f}\n
Train: train_loss {np.mean(losses):.3f}\n
Val: val_loss {val_loss:.3f} val_auc {epoch_auc:.3f} \n''')


def compare_auc(epoch_auc, prev_val_auc, lr_scheduler,
                model, output_file):
    if epoch_auc > prev_val_auc:
        prev_val_auc = epoch_auc
        tag = lr_scheduler if lr_scheduler else ''
        path = f"model_{tag}_auc_{100*epoch_auc:.0f}.pth\n"
        save_model(model, path)
        with open(output_file, 'a') as file:
            file.write(f'''Model: {path}''')
    return prev_val_auc


def train_mura_model(model: nn.Module, optimizer: torch.optim.Adam,
                     train_dl, valid_dl, epochs: int = 25, track_loss=False,
                     lr_scheduler=None, criterion: torch.nn.Module = MURALoss,
                     args=None, wandb_proj: str = 'mura_ml', **kwargs):
    wandb.init(project=wandb_proj, config=locals())
    log_dir = '/home/cwbennie/MURA_logs/'
    output_file = f'{log_dir}{args.log_file}.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer(parameters, lr=args.learning_rate,
                          betas=(0.9, 0.999))
    if lr_scheduler is not None:
        # TODO need to check baselines with repo
        if not kwargs.get('step_size'):
            kwargs['step_size'] = epochs // 3
        if not kwargs.get('gamma'):
            kwargs['gamma'] = 0.95
        if not kwargs.get('max_lr'):
            kwargs['max_lr'] = args.learning_rate * 2.5
        if not kwargs.get('steps_per_epoch'):
            kwargs['steps_per_epoch'] = len(train_dl)
        if not kwargs.get('epochs'):
            kwargs['epochs'] = epochs
        scheduler = get_scheduler(name=lr_scheduler, optimizer=optimizer,
                                  **kwargs)
        output_file = f'{log_dir}{lr_scheduler}_{args.log_file}.txt'
    epoch_losses, auc_scores, val_losses = list(), list(), list()
    criterion = criterion.to(device)
    prev_val_auc = 0.0

    # train the model for the given number of epochs
    for i in range(epochs):
        start = time.time()
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
            val_loss, auc, preds, y_val = mura_model_eval(model, valid_dl,
                                                          criterion=criterion,
                                                          test=True)
            val_losses.append(val_loss)
            auc_scores.append(auc)
        elapsed = time.time() - start
        log_train(output_file, elapsed, losses,
                  val_loss, auc, epoch_num=i)
        wandb.log({'Epoch Loss': np.mean(losses),
                   'Validation Loss': val_loss,
                   'Validation AUC': auc})
        prev_val_auc = compare_auc(auc, prev_val_auc, lr_scheduler,
                                   model, output_file)
        if lr_scheduler is not None:
            scheduler.step()
    wandb.log({'roc': wandb.plot.roc_curve(y_val, preds,
                                           labels=None,
                                           classes_to_plot=None)})
    wandb.finish()
    if track_loss:
        pkl_file = output_file.strip('.txt') + '.pkl'
        with open(pkl_file, 'wb') as file:
            pickle.dump((epoch_losses, val_losses, auc_scores, preds, y_val),
                        file=file)
        return (epoch_losses, val_losses, auc_scores, preds, y_val)


def train_triangular_policy(model, train_dl, valid_dl, criterion,
                            max_lr=0.04, epochs=5, args=None,
                            wandb_proj: str = 'mura_ml',
                            track_loss=False):
    wandb.init(project=wandb_proj, config=locals())
    output_file = f'/home/cwbennie/MURA_logs/triangular_{args.log_file}.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    idx = 0
    iterations = epochs*len(train_dl)
    lrs = get_cosine_triangular_lr(max_lr, iterations)
    optimizer = create_optimizer(model, lrs[0])
    prev_val_auc = 0.0
    epoch_losses, auc_scores, val_losses = list(), list(), list()
    for j in range(epochs):
        start = time.time()
        model.train()
        losses = list()
        for i, (img, y, wts) in enumerate(train_dl):
            lr = lrs[idx]
            update_optimizer(optimizer, [lr/9, lr/3, lr])
            img = img.to(device).float()
            y = y.to(device).float()
            out = model(img)
            loss = criterion(out.squeeze(), y, wts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            idx += 1
        epoch_losses.append(np.mean(losses))
        if valid_dl:
            val_loss, auc, preds, y_val = mura_model_eval(model, valid_dl,
                                                          criterion=criterion,
                                                          test=True)
            val_losses.append(val_loss)
            auc_scores.append(auc)
        elapsed = time.time() - start
        wandb.log({'Epoch Loss': np.mean(losses),
                   'Validation Loss': val_loss,
                   'Validation AUC': auc})
        log_train(output_file, elapsed, losses, val_loss, auc,
                  epoch_num=i)
        prev_val_auc = compare_auc(auc, prev_val_auc, None,
                                   model, output_file)
    wandb.log({'roc': wandb.plot.roc_curve(y_val, preds,
                                           labels=None,
                                           classes_to_plot=None)})
    wandb.finish()
    if track_loss:
        pkl_file = output_file.strip('.txt') + '.pkl'
        with open(pkl_file, 'wb') as file:
            pickle.dump((epoch_losses, val_losses, auc_scores, preds, y_val),
                        file=file)
        return (epoch_losses, val_losses, auc_scores, preds, y_val)


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


# Begin Functions for CheXpert
def get_chexpert_labels(path: Path, dataframe: pd.DataFrame,
                        data_type: str = 'train'):
    file_name = str(path).split('/')[6:]
    file_name = '/'.join(file_name)
    full_id = f'CheXpert-v1.0/{data_type}/' + file_name
    return torch.Tensor(list(map(int,
                                 list(dataframe.loc[full_id].values)[4:])))


def get_chexpert_images(path: Path, label_csv: str,
                        data_type: str = 'train',
                        uncertainty_type: str = 'own_class'):
    images = list()
    label_info = pd.read_csv(label_csv)
    label_info.fillna(0.0, inplace=True)
    if uncertainty_type == 'own_class':
        label_info.replace(-1, 2, inplace=True)
    elif uncertainty_type == 'pos_replace':
        label_info.replace(-1, 1, inplace=True)
    else:
        label_info.replace(-1, 0, inplace=True)
    label_info.set_index('Path', inplace=True)
    labels = list()
    for dirpath, dirnames, files in os.walk(Path(path)):
        for file in files:
            if file.endswith('jpg'):
                im_file = os.path.join(dirpath, file)
                images.append(im_file)
                label = get_chexpert_labels(im_file, label_info, data_type)
                labels.append(label)
    return images, labels


def plot_auc(roc_data, mod_title, save_path):
    pred_scores, true_labels = roc_data
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{mod_title} - ROC Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(os.path.join(save_path, f"{mod_title}_ROC.png"))
    plt.show()


def plot_losses(losses, mod_title, save_path):
    # Print and show the loss curves and AUC
    labels = ['Training Loss', 'Validation Loss']
    for loss, label in zip(losses, labels):
        plt.plot(loss, label=label)
        plt.title(f'{mod_title} - {label}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if label == 'Validation Loss':
            plt_title = f"{mod_title}_{label.replace(' ', '_')}.png"
            plt.savefig(os.path.join(save_path, plt_title))
        plt.show()


def get_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
    elif model_name == 'resnet34':
        model = models.resnet34(weights='DEFAULT')
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
    elif model_name == 'mobilenet_large':
        model = models.mobilenet_v3_large(weights='DEFAULT')
    elif model_name == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
    elif model_name == 'densenet161':
        model = models.densenet161(weights='DEFAULT')
    elif model_name == 'densenet169':
        model = models.densenet169(weights='DEFAULT')
    return model


def get_base(model_name: str):
    model = get_model(model_name)
    if model_name.startswith('densenet'):
        layers = list(model.children())[0]  # check if this holds up for all densenets
    elif model_name.startswith('resnet'):
        layers = list(model.children())[:8]  # need to check to see if this holds up for all resnets
    elif model_name.startswith('mobilenet'):
        layers = nn.Sequential(model.features)
    return layers


def get_classifier(model_name: str, output: int):
    if model_name.startswith('densenet'):
        classifer = nn.Linear(1024, output)
    elif model_name.startswith('resnet'):
        classifer = nn.Linear(512, output)
    elif model_name.startswith('mobilenet'):
        classifer = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, output)
        )
    return classifer


def get_layers(model_name, output):
    mod_names = ['resnet18', 'resnet34', 'resnet50', 'mobilenet_large',
                 'densenet121', 'densenet161', 'densenet169']
    if model_name not in mod_names:
        model_name = 'densenet121'
    base_model = get_base(model_name)
    classifier = get_classifier(model_name, output)
    global_pool = nn.AdaptiveAvgPool2d((1, 1))
    return base_model, classifier, global_pool


def get_groups(base_layers, classifier):
    # need to check if we can break layers up into 3 for resnet and mobilenet
    groups = nn.ModuleList([nn.Sequential(*h) for h in
                            [base_layers[:7], base_layers[7:]]])
    groups.append(classifier)
    return groups
