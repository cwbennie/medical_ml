import argparse
from pathlib import Path
from datasets import MuraData
from models import MURANet, MURALoss
from utils import train_triangular_policy, train_mura_model, plot_losses, plot_auc

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
                    help='set the maximum learning rate for triangular training')
parser.add_argument('--log_file', required=True, type=str,
                    help='add a suffix for the log file for the training')
parser.add_argument('--img_dir', required=True, type=str,
                    help='directory to save plot images from training')

args = parser.parse_args()

def main():
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
                                          args=args)
    else:
        results = train_mura_model(model=model, train_dl=train_dl,
                                   valid_dl=test_dl, epochs=args.epochs,
                                   criterion=loss, lr_scheduler=args.train_method,
                                   track_loss=True, args=args)
    
    losses, val_losses, auc_scores, preds, y_val = results
    plot_losses([losses, val_losses, auc_scores],
             mod_title=str(args.baseline_model).capitalize(),
             save_path=args.img_dir)
    plot_auc((preds, y_val), mod_title=str(args.baseline_model).capitalize(),
             save_path=args.img_dir)


if __name__ == '__main__':
    main()