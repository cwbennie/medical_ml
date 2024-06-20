import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2


def get_moco_images(path: Path):
    images = list()
    for dirpath, dirnames, files in os.walk(Path(path)):
        for file in files:
            if file.endswith('jpg'):
                im_file = os.path.join(dirpath, file)
                images.append(im_file)
    return images


def transform_image(image, transform_pipe: A.Compose):
    img = cv2.imread(str(image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform_pipe(image=img)
    return augmented['image']


class MoCoData(Dataset):
    def __init__(self, dir_path: Path, transforms: A.Compose = None) -> None:
        self.image_dir = dir_path
        self.images = get_moco_images(self.image_dir)
        self.transforms = transforms
        if not transforms:
            self.transforms = A.Compose([
                A.RandomResizedCrop((224, 224), scale=(0.2, 1)),
                A.ColorJitter(0.4, 0.4, 0.4, 0.4),
                A.HorizontalFlip(),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        im_q = transform_image(img, self.transforms)
        im_k = transform_image(img, self.transforms)
        return (im_q, im_k)


class BasicMoco(nn.Module):
    """
    Construct a basic MoCo model with:
    query encoder, key encoder, and queue.
    """
    def __init__(self, base_encoder: nn.Module, dim: int = 128,
                 queue_size: int = 1000, momentum: float = 0.999,
                 temp: float = 0.07, mlp: bool = False,
                 pretrained: bool = False):
        """
        dim: feature dimension (default: 128)
        queue size: the number of negative keys for contrastive
            learning (default: 65536)
        momentum: moco momentum of updating key encoder (default: 0.999)
        temp: softmax temperature (default: 0.07)
        """
        super(BasicMoco, self).__init__()

        self.K = queue_size
        self.m = momentum
        self.T = temp

        if pretrained:
            model_options = ['resnet', 'densnet']

            self.encoder_q = base_encoder(weights='DEFAULT')

            if self.encoder_q.__class__.__name__.lower() not in model_options:
                print('Model type not supported - resroting to Resnet18')
                base_encoder = models.resnet18
                self.encoder_q = base_encoder(weights='DEFAULT')

            if self.encoder_q.__class__.__name__.lower() == 'resnet':
                self.encoder_k = base_encoder(weights='DEFAULT')
                num_ftrs = self.encoder_q.fc.in_features
                self.encoder_q.fc = nn.Linear(num_ftrs, dim)
                self.encoder_k.fc = nn.Linear(num_ftrs, dim)
            elif self.encoder_q.__class__.__name__.lower() == 'densenet':
                self.encoder_k = base_encoder(weights='DEFAULT')
                num_ftrs = self.encoder_q.classifier.in_features
                self.encoder_q.classifier = nn.Linear(num_ftrs, dim)
                self.encoder_k.classifier = nn.Linear(num_ftrs, dim)

        # if we want a final MLP layer, add an additional layer
        if mlp:
            if self.encoder_q.__class__.__name__.lower() == 'resnet':
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)
            elif self.encoder_q.__class__.__name__.lower() == 'densenet':
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.ReLU(),
                    self.encoder_q.classifier)
                self.encoder_k.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.ReLU(),
                    self.encoder_k.classifier)

        # ensure the key encoder parameters are the same as the query encoder
        # and set the key encoder parameter gradients to false
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data._copy(param_q.data)
            param_k.requires_grad = False

        # intialize queue structure to be used in training
        # queue should be of size dim x K
        self.register_buffer('queue', torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # create buffer structure to track the pointer for the queue
        self.register_buffer('queue_ptr', torch.zeros_like(torch.empty(1),
                                                           dtype=torch.long))

        def _momentum_update_key_encoder(self: BasicMoco):
            """
            Function to update key encoder weights based on updates to query
            encoder and using momentum parameter
            """
            for param_q, param_k in zip(self.encoder_q.parameters(),
                                        self.encoder_k.parameters()):
                query_enc_update = param_q.data * (1 - self.m)
                key_enc_update = param_k.data * self.m
                param_k.data = key_enc_update + query_enc_update

        def _dequeue_and_enqueu(self: BasicMoco, keys: torch.Tensor):
            batch_size = keys.shape[0]

            # get the pointer location to update keys
            ptr = int(self.queue_ptr)
            assert self.K % batch_size == 0

            # replace selected negative samples in queue
            self.queue[:, ptr:ptr + batch_size] = keys.T

            # move pointer for next sample
            ptr = (ptr + batch_size) % self.K
            self.queue_ptr[0] = ptr

        def forward(self: BasicMoco, im_q: torch.Tensor, im_k: torch.Tensor):
            """
            Input:
                im_q: a batch of query images
                im_k: a batch of key images

            Output:
                logits, targets
            """

            # compute query features with query encoder
            q = self.encoder_q(im_q)
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            # ensure gradients are not tracked as we update using the
            # gradients from the query encoder
            with torch.no_grad():
                # update key encoder gradients
                self._momentum_update_key_encoder()

                # get key features
                k = self.encoder_k(im_k)
                k = nn.functional.normalize(k, dim=1)

            # compute logits
            # positive logits - size Nx1
            # unsqueeze to add column dimension to results
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

            # negative logits - size NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # combine one positive logit and negative logits
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature to logits
            logits /= self.T

            # create tensor for labels
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue using new key encodings
            self._dequeue_and_enqueue(k)

            return logits, labels


## TODO Finish training loop
def moco_train(model: nn.Module, epochs: int=10,
               train_dl: DataLoader=None, valid_dl: DataLoader=None,
               ):

    for i in range(epochs):
        pass
