#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：financial_ner 
# @File    ：rdrop_loss.py
# @Author  ：sl
# @Date    ：2022/3/21 14:31


import torch
import torch.nn as nn
import torch.nn.functional as F

"""
R-Drop: Regularized Dropout for Neural Networks
"""


class RDropLoss(nn.Module):
    """
    配合R-Drop的交叉熵损失
    """

    def __init__(self, alpha=4, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss()
        self.kld = torch.nn.KLDivLoss(reduction=reduction)
        # self.kld = torch.nn.KLDivLoss()

    def forward(self, y_pred, labels=None, ce_loss=None):
        """
        计算 loss
            配合上述生成器的R-Drop Loss
            其实loss_kl的除以4，是为了在数量上对齐公式描述结果。
        :param y_pred:
        :param labels:
        :param ce_loss:
        :return:
        """
        if ce_loss is None:
            loss1 = self.ce(y_pred, labels)
        else:
            loss1 = ce_loss

        y_pred1 = y_pred[::2]
        y_pred2 = y_pred[1::2]
        loss_kl1 = self.kld(torch.log_softmax(y_pred1, dim=1), y_pred2.softmax(dim=-1))
        loss_kl2 = self.kld(torch.log_softmax(y_pred2, dim=1), y_pred1.softmax(dim=-1))

        loss2 = loss_kl1 + loss_kl2
        loss2_mean = torch.mean(loss2)
        loss = loss1 + loss2_mean / 4 * self.alpha

        # loss2 = self.kld(torch.log_softmax(y_pred[::2], dim=1), y_pred[1::2].softmax(dim=-1)) + \
        #         self.kld(torch.log_softmax(y_pred[1::2], dim=1), y_pred[::2].softmax(dim=-1))
        #
        # loss = loss1 + torch.mean(loss2) / 4 * self.alpha
        return loss
