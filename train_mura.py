import os
import argparse
from pathlib import Path
from datasets import MuraData
from models import MURANet, MURALoss
from torch.optim import Adam
from utils import (train_triangular_policy, train_mura_model,
                   plot_losses, plot_auc)

parser = argparse.ArgumentParser(description="MURA X-Ray Training")
parser.add_argument('--baseline_model', type=str, required=True,
                    help='which model to use in training',
                    choices=['resnet18', 'resnet34',
                             'resnet50', 'mobilenet_large',
                             'densenet121', 'densenet161', 'densenet169'])
parser.add_argument('--train_method', type=str, required=True,
                    help='select which kind of training method to use',
                    choices=['triangular', 'one_cycle', 'step_decay',
                             'exp_decay'])
parser.add_argument('--train_batch', type=int, default=40,
                    help='default is set to 40 to avoid CUDA memory issues')
parser.add_argument('--epochs', type=int, default=10,
                    help='default set to 10')
parser.add_argument('--image_size', type=int, default=320,
                    help='default is set to 320')
parser.add_argument('--triangular_lr', type=float, default=0.001,
                    help='set the max learning rate for triangular training')
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='set a learning rate for training methods')
parser.add_argument('--log_file', required=True, type=str,
                    help='add a suffix for the log file for the training')
parser.add_argument('--img_dir', required=True, type=str,
                    help='directory to save plot images from training')
parser.add_argument('--gpu', type=int, default=0,
                    help='select which GPU device to use [0-7]')
parser.add_argument('--wandb_proj', type=str, default='mura_ml',
                    help='name of weights and biases project for logging')

args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    train_pth = Path('/home/cwbennie/data/MURA-v1.1/train')
    mura_train = MuraData(dir_path=train_pth, transform=True,
                          img_size=args.image_size)
    train_dl = mura_train.train_dataloader(batch_size=args.train_batch)

    val_pth = Path('/home/cwbennie/data/MURA-v1.1/valid')
    mura_test = MuraData(dir_path=val_pth, transform=False,
                         img_size=args.image_size)
    test_dl = mura_test.test_dataloader()

    model = MURANet(args=args)
    loss = MURALoss()
    if args.train_method == 'triangular':
        results = train_triangular_policy(model, train_dl, test_dl,
                                          criterion=loss,
                                          max_lr=args.triangular_lr,
                                          epochs=args.epochs, track_loss=True,
                                          wandb_proj=args.wandb_proj,
                                          args=args)
    else:
        results = train_mura_model(model=model, train_dl=train_dl,
                                   optimizer=Adam,
                                   valid_dl=test_dl, epochs=args.epochs,
                                   criterion=loss,
                                   lr_scheduler=args.train_method,
                                   wandb_proj=args.wandb_proj,
                                   track_loss=True, args=args)

    losses, val_losses, auc_scores, preds, y_val = results
    save_pth = '/home/cwbennie/MURA_imgs/'
    plot_losses([losses, val_losses],
                mod_title=str(args.baseline_model).capitalize(),
                save_path=save_pth)
    plot_auc((preds, y_val), mod_title=str(args.baseline_model).capitalize(),
             save_path=save_pth)


if __name__ == '__main__':
    main()
