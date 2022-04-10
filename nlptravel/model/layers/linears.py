#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel
# @File    ：linears.py
# @Author  ：sl
# @Date    ：2022/3/9 16:57

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPooler


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


class MultiLayerPerceptronClassifier(nn.Module):
    def __init__(self, hidden_size=None, num_labels=None, activate_func="gelu"):
        super().__init__()
        self.dense_layer = nn.Linear(hidden_size, hidden_size)
        self.dense_to_labels_layer = nn.Linear(hidden_size, num_labels)
        if activate_func == "tanh":
            self.activation = nn.Tanh()
        elif activate_func == "relu":
            self.activation = nn.ReLU()
        elif activate_func == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError

    def forward(self, sequence_hidden_states):
        sequence_output = self.dense_layer(sequence_hidden_states)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dense_to_labels_layer(sequence_output)
        return sequence_output


class SpanClassifier(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float):
        super(SpanClassifier, self).__init__()
        self.start_proj = nn.Linear(hidden_size, hidden_size)
        self.end_proj = nn.Linear(hidden_size, hidden_size)
        self.biaffine = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.concat_proj = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def forward(self, input_features):
        bsz, seq_len, dim = input_features.size()
        # B, L, h
        start_feature = self.dropout(F.gelu(self.start_proj(input_features)))
        # B, L, h
        end_feature = self.dropout(F.gelu(self.end_proj(input_features)))
        # B, L, L
        biaffine_logits = torch.bmm(torch.matmul(start_feature, self.biaffine), end_feature.transpose(1, 2))

        start_extend = start_feature.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [B, L, L, h]
        end_extend = end_feature.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [B, L, L, h]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [B, L, L]
        concat_logits = self.concat_proj(span_matrix).squeeze(-1)
        # B, L, L
        return biaffine_logits + concat_logits

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.biaffine, a=math.sqrt(5))


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class BertMLP(nn.Module):
    def __init__(self, config, ):
        super().__init__()
        self.dense_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_to_labels_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.activation = nn.Tanh()

    def forward(self, sequence_hidden_states):
        sequence_output = self.dense_layer(sequence_hidden_states)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dense_to_labels_layer(sequence_output)
        return sequence_output


class LmPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0., add_pooling_layer=True):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        self.pooler = LmPooler(input_dim)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, hidden_states):
        # first_token_tensor = hidden_states[:, 0]
        pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(pooled_output)
        return self.linear(pooled_output)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
