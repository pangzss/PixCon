# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.utils.logging import logger_initialized, print_log

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel

from einops import rearrange

@ALGORITHMS.register_module()
class PixCon(BaseModel):
    """DenseCL.

    Implementation of `Dense Contrastive Learning for Self-Supervised Visual
    Pre-Training <https://arxiv.org/abs/2011.09157>`_.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL>`_.
    The loss_lambda warmup is in `core/hooks/densecl_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
        loss_lambda (float): Loss weight for the single and dense contrastive
            loss. Defaults to 0.5.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 predictor=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 loss_lambda=0.5,
                 init_cfg=None,
                 sr=True,
                 use_mask=True,
                 power=2,
                 **kwargs):
        super(PixCon, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        self.backbone = self.encoder_q[0]
        assert head is not None
        self.predictor = build_neck(predictor)
        self.predictor_dense = build_neck(predictor)
        self.head = build_head(head)

        self.queue_len = queue_len
        self.base_momentum = momentum
        self.momentum = momentum
        self.loss_lambda = loss_lambda

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        # create the second queue for dense output
        self.register_buffer('queue2', torch.randn(feat_dim, queue_len))
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.register_buffer('queue2_ptr', torch.zeros(1, dtype=torch.long))

        self.sr = sr
        self.use_mask = use_mask 
        self.power = power
    def init_weights(self):
        """Init weights and copy query encoder init weights to key encoder."""
        super().init_weights()

        # Get the initialized logger, if not exist,
        # create a logger named `mmselfsup`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmselfsup'

        # log that key encoder is initialized by the query encoder
        print_log(
            'Key encoder is initialized by the query encoder.',
            logger=logger_name)

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        """Update queue2."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x
    
    def calc_dense_loss(self, q_b, k_b, mask_q):
        mask_q = mask_q.view(mask_q.shape[0], -1)
        # compute query features
        q_grid, _ = self.encoder_q[1]([q_b], "dense")  # queries: NxC; NxCxS^2
        q_grid = rearrange(q_grid, "b c m -> (b m) c")
        q_grid = self.predictor_dense([q_grid])[0]
        q_grid = F.normalize(rearrange(q_grid, "(b m) c -> b c m", b=q_b.shape[0]), dim=1)

        q_b_norm = F.normalize(q_b.view(q_b.size(0), q_b.size(1), -1), dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            k_grid, k2 = self.encoder_k[1]([k_b], "dense")  # keys: NxC; NxCxS^2
            k_grid = F.normalize(k_grid, dim=1)
            k2 = F.normalize(k2, dim=1)
            
            k_b_norm = F.normalize(k_b.view(k_b.size(0), k_b.size(1), -1), dim=1)
        # feat point set sim
        with torch.no_grad():
            backbone_sim_matrix = torch.matmul(q_b_norm.permute(0, 2, 1), k_b_norm) # NxS^2xS^2

        weights, densecl_sim_ind = backbone_sim_matrix.max(dim=2)  # NxS^2

        indexed_k_grid = torch.gather(k_grid, 2,
                                      densecl_sim_ind.unsqueeze(1).expand(
                                          -1, k_grid.size(1), -1))  # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # NxS^2

        # dense positive logits: NS^2X1
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        # dense negative logits: NS^2xK
        l_neg_dense = torch.einsum(
            'nc,ck->nk', [q_grid, self.queue2.clone().detach()])

        if self.sr:  
            if self.use_mask:
                weights = (weights - weights.min(dim=1, keepdim=True)[0]) / (weights.masked_fill(mask_q==1, -2).max(dim=1, keepdim=True)[0] - weights.min(dim=1, keepdim=True)[0])
                weights = weights.masked_fill(mask_q == 1, 1.)
            weights = weights ** self.power
            weights_flat = weights.view(-1) 

        logits = torch.cat((l_pos_dense, l_neg_dense), dim=1)
        logits /= self.head.temperature
        labels = torch.zeros((logits.shape[0], ), dtype=torch.long).to(logits.device)
        loss_dense = F.cross_entropy(logits, labels, reduction="none")
        if self.sr:
            loss_dense = (loss_dense * weights_flat).sum() / weights_flat.sum()
        else: 
            loss_dense = loss_dense.mean()

        loss_dense = loss_dense * self.loss_lambda
        return loss_dense, k2
    
    def calc_image_loss(self, q_b, k_b):
        q, _ = self.encoder_q[1]([q_b], "image") 
        q = F.normalize(self.predictor([q])[0], dim=1)
        with torch.no_grad(): 
            k, k_avg = self.encoder_k[1]([k_b], "image")
            
            k = F.normalize(k, dim=1)
            k_avg = F.normalize(k_avg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        loss_single = self.head(l_pos, l_neg)['loss']
        loss_single = loss_single * (1 - self.loss_lambda)
        return loss_single, k_avg
     
    def forward_train(self, img, masks, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        q_b_v1 = self.encoder_q[0](img[0])[0]
        q_b_v2 = self.encoder_q[0](img[1])[0]
        with torch.no_grad():
            k_b_v1 = self.encoder_k[0](img[0])[0]
            k_b_v2 = self.encoder_k[0](img[1])[0] 

        loss_dense_v1, k2_v2 = self.calc_dense_loss(q_b_v1, k_b_v2, masks[0])
        loss_dense_v2, k2_v1 = self.calc_dense_loss(q_b_v2, k_b_v1, masks[1])

        loss_single_v1, k_v2 = self.calc_image_loss(q_b_v1, k_b_v2)
        loss_single_v2, k_v1 = self.calc_image_loss(q_b_v2, k_b_v1)

        losses = dict()
        losses["loss_single"] = loss_single_v1 + loss_single_v2
        losses["loss_dense"] = loss_dense_v1 + loss_dense_v2
        
        if np.random.rand() < 0.5:
            k_to_enqueue = k_v2
            k2_to_enqueue = k2_v2
        else:
            k_to_enqueue = k_v1
            k2_to_enqueue = k2_v1
        self._dequeue_and_enqueue(k_to_enqueue)
        self._dequeue_and_enqueue2(k2_to_enqueue)
        return losses

    def forward_test(self, img, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input of two concatenated images of shape
                (N, 2, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict(Tensor): A dictionary of normalized output features.
        """
        im_q = img.contiguous()
        # compute query features
        # _, q_grid, _ = self.encoder_q(im_q)
        q_grid = self.extract_feat(im_q)[0]
        q_grid = q_grid.view(q_grid.size(0), q_grid.size(1), -1)
        q_grid = F.normalize(q_grid, dim=1)
        return None, q_grid, None
