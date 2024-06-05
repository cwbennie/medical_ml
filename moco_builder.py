# Based on work complete by Facebook Research team
# https://github.com/facebookresearch/moco
import torch
import torch.distributed
import torch.nn as nn
from torchvision import models


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    """
    Build a MoCo model consisting of: query encoder, key encoder,
    and queue.
    """

    def __init__(self, base_encoder: nn.Module, dim=128, K=65536,
                 m=0.999, T=0.07, mlp=False, pretrained=False):
        """
        dim: feature dimension (default: 128)
        K: queue size - the number of negative keys for contrastive
            learning (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        if pretrained:

            pt_models = ['resnet', 'densenet']

            self.encoder_q = base_encoder(weights='DEFAULT')

            if self.encoder_q.__class__.__name__.lower() not in pt_models:
                print('Model type not supported - resorting to Resnet18')
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

        # add additional layer for final MLP
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

        # we need to set the key encoder parameters gradients to False
        # MoCO updates this through the momentum update function below
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data._copy(param_q.data)
            param_k.requires_grad = False

        # initialize the queue structure
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer('queue_ptr', torch.zeros_like(torch.empty(1),
                                                           dtype=torch.long))

        @torch.no_grad()
        def _momentum_update_key_encoder(self):
            """
            MoCo function using momentum to update the key encoder weights
            """
            for param_q, param_k in zip(self.encoder_q.parameters(),
                                        self.encoder_k.parameters()):
                query_enc_update = param_q.data * (1.0 - self.m)
                key_enc_update = param_k.data * self.m
                param_k.data = query_enc_update + key_enc_update

        @torch.no_grad()
        def _dequeue_and_enqueue(self, keys):
            # gather keys before updating queue
            keys = concat_all_gather(keys)

            batch_size = keys.shape[0]

            ptr = int(self.queue_ptr)
            assert self.K % batch_size == 0

            # replace keys at ptr - dequeue and enqueue step
            self.queue[:, ptr:ptr + batch_size] = keys.T
            # move pointer for next iteration
            ptr = (ptr + batch_size) % self.K

            self.queue_ptr[0] = ptr

        @torch.no_grad()
        def _batch_shuffle_ddp(self, x):
            """
            Batch shuffle to make use of BatchNorm.
            *** To be used with DistributedDataParallel model ***
            """
            # gather tensors from all gpus
            batch_size_this = x.shape[0]
            x_gather = concat_all_gather(x)
            batch_size_all = x_gather.shape[0]
            num_gpus = batch_size_all // batch_size_this

            # random shuffle index
            idx_shuffle = torch.randperm(batch_size_all).cuda()

            # broadcast to all gpus
            torch.distributed.broadcast(idx_shuffle, src=0)

            # index for restoring (to be used in unshuffle function)
            idx_unshuffle = torch.argsort(idx_shuffle)

            # shuffled index for this gpu
            gpu_idx = torch.distributed.get_rank()
            idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

            return x_gather[idx_this], idx_unshuffle

        @torch.no_grad()
        def _batch_unshuffle_ddp(self, x, idx_unshuffle):
            """
            Undo batch shuffle.
            *** To be used with DistributedDataParallel model ***
            """
            # gather from all gpus
            batch_size_this = x.shape[0]
            x_gather = concat_all_gather(x)
            batch_size_all = x_gather.shape[0]

            num_gpus = batch_size_all // batch_size_this

            # restored index for this gpu
            gpu_idx = torch.distributed.get_rank()
            idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

            return x_gather[idx_this]

        def forward(self, im_q, im_k):
            """
            Input:
                im_q: a batch of query images
                im_k: a batch of key images

            Output:
                logits, targets
            """

            # compute queryfeatures
            q = self.encoder_q(im_q)
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            # we do this without tracking gradients as we use query
            # gradients to update the key encoder
            with torch.no_grad():
                # update the key encoder gradients
                self._momentum_update_key_encoder()

                # shuffle to make use of Batch Normalization
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.encoder_k(im_k)
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # compute logits
            # Original repo uses Einstein sum
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+L)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            return logits, labels
