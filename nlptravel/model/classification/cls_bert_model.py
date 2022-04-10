#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel
# @File    ：cls_bert_model.py
# @Author  ：sl
# @Date    ：2022/3/9 16:57
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, ElectraModel, ElectraPreTrainedModel

from nlptravel.loss.focal_loss import FocalLoss
from nlptravel.loss.label_smoothing import LabelSmoothingCrossEntropy
from nlptravel.model.layers.crf import CRF
from nlptravel.model.layers.linears import IntentClassifier, SlotClassifier


class ClsBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        bert_output = self.bert(context, attention_mask=mask)
        out = self.fc(bert_output.pooler_output)
        return out


class ClsBertModelV2(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel.from_pretrained(config.bert_path)
        self.fc = torch.nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.bert(input_ids, attention_mask, token_type_ids)
        x = x[0][:, 0, :]  # 取cls向量
        x = self.fc(x)
        return x


##################################################################################
# JointBert
#  - BERT for Joint Intent Classification and Slot Filling
#
# 2022-03-30
##################################################################################

"""
JointBert 分类模型
"""


# class JointBERT(BertPreTrainedModel):
class JointBERT(ElectraPreTrainedModel):
    def __init__(self, config, args=None, num_intent_labels=2, num_slot_labels=2,
                 dropout_rate=0.1, slot_loss_coef=1.0):
        super().__init__(config)
        self.args = args
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels

        # self.use_crf = args.use_crf
        self.use_crf = False
        self.dropout_rate = dropout_rate
        self.slot_loss_coef = slot_loss_coef

        # self.bert = BertModel(config=config)
        self.bert = ElectraModel(config=config)

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, self.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, self.dropout_rate)

        if self.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.loss_type = "ce"
        self.loss_fct = None
        self.build_criterion()

        self.init_weights()

    def build_criterion(self):
        """
        获取loss 类型
        """
        assert self.loss_type in ['lsr', 'focal', 'ce']
        if self.loss_type == 'lsr':
            self.loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
        elif self.loss_type == 'focal':
            self.loss_fct = FocalLoss(ignore_index=0)
        else:
            self.loss_fct = CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids=None,
                attention_mask=None,
                loss_mask=None,
                src_lens=None,
                token_type_ids=None,
                labels=None,
                detect_labels=None):
        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        # pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        loss = None
        # 1. Intent Softmax
        if labels is not None:

            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), labels.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), labels.view(-1))
            loss = intent_loss

        # 2. Slot Softmax
        if detect_labels is not None:
            if self.use_crf:
                slot_loss = self.crf(slot_logits, detect_labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = self.loss_fct
                # Only keep active parts of the loss
                if loss_mask is not None:
                    active_loss = loss_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = detect_labels.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), detect_labels.view(-1))
            loss = loss + self.slot_loss_coef * slot_loss

        if loss is not None:
            outputs = (loss,) + outputs

        # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
        return outputs
