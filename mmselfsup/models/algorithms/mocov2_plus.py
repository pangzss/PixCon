# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class MoCoV2plus(BaseModel):
    """MoCoV2+.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

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
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 predictor=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.99,
                 init_cfg=None,
                 **kwargs):
        super(MoCoV2plus, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        self.predictor = build_neck(predictor)
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.base_momentum = momentum
        self.momentum = momentum

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

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

    def forward_train(self, img, **kwargs):
        """Forward computation during training.
           Symmetrize loss.
        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        im_v1 = img[0]
        im_v2 = img[1]
        # compute query features
        q_v1 = self.predictor(self.encoder_q(im_v1))[0]  # queries: NxC
        q_v2 = self.predictor(self.encoder_q(im_v2))[0]  # queries: NxC
        q_v1 = nn.functional.normalize(q_v1, dim=1)
        q_v2 = nn.functional.normalize(q_v2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            # self._momentum_update_key_encoder()
            # shuffle bn removed as syncbn is used.
            k_v1 = self.encoder_k(im_v1)[0]  # keys: NxC
            k_v2 = self.encoder_k(im_v2)[0]  # keys: NxC
            k_v1 = nn.functional.normalize(k_v1, dim=1)
            k_v2 = nn.functional.normalize(k_v2, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_v1 = torch.einsum('nc,nc->n', [q_v1, k_v2]).unsqueeze(-1)
        l_pos_v2 = torch.einsum('nc,nc->n', [q_v2, k_v1]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_v1 = torch.einsum('nc,ck->nk', [q_v1, self.queue.clone().detach()])
        l_neg_v2 = torch.einsum('nc,ck->nk', [q_v2, self.queue.clone().detach()])

        losses_v1 = self.head(l_pos_v1, l_neg_v1)
        losses_v2 = self.head(l_pos_v2, l_neg_v2)

        # update the queue
        if np.random.rand() < 0.5:
            k_to_enqueue = k_v1
        else:
            k_to_enqueue = k_v2 
        self._dequeue_and_enqueue(k_to_enqueue)

        losses = dict()
        losses["loss"] = losses_v1["loss"] + losses_v2["loss"]
        return losses 
