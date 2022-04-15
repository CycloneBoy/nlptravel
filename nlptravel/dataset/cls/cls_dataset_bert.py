#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project nlptravel
# @File  : cls_dataset_bert.py
# @Author: sl
# @Date  : 2022/4/10 - 下午2:34


import os

from typing import List, Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import random
import pandas as pd

from transformers import PreTrainedTokenizer

from nlptravel.utils.base_utils import DataFileType
from nlptravel.utils.constant import Constants
from nlptravel.utils.file_utils import FileUtils
from nlptravel.utils.logger_utils import logger

"""

CLS

PLOME: Pre-training with Misspelled Knowledge
for Chinese Spelling Correction

"""


class ClsBertDataset(Dataset):

    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 block_size: int = 180,
                 need_cache=False, input_examples=None, dataset_name="cls", use_rdrop_loss=False):

        # print(file_path, os.path.isfile(file_path))
        assert os.path.isfile(file_path)

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_sen_len = block_size
        self.need_cache = need_cache
        self.data_type = DataFileType.CSV
        self.dataset_name = dataset_name
        self.use_rdrop_loss = use_rdrop_loss

        # 标签 "tnews_public"
        self.labels = ["0", "1"]
        self.label2id = {}
        self.id2label = {}
        self.make_label()

        # 是否从catch 加载
        self.cache_feature_file = file_path[:-4] + '_cache.pth'

        self.input_examples = [] if input_examples is None else input_examples

        self.input_ids = []
        self.label_ids = []

        if os.path.exists(self.cache_feature_file) and self.need_cache:
            logger.info(f"find data features cache : {self.cache_feature_file}")
            self.load_cache_feature()
        else:
            self.read_data_from_file()
            if self.need_cache:
                self.cache_feature()

    def make_label(self):
        if self.dataset_name == Constants.DATASET_NAME_CLS_TNEWS:
            self.labels = Constants.DATASET_NAME_CLS_TNEWS_LABELS

            self.data_type = DataFileType.JSON_LINE

        for index, item in enumerate(self.labels):
            self.label2id[item] = index
            self.id2label[index] = item

    def read_data_from_file(self):
        if len(self.input_examples) == 0:
            self.input_examples = FileUtils.read_data_from_file(self.file_path, data_type=self.data_type,
                                                                column_name=Constants.COLUMN_NAME_2)
            logger.info("Creating features from dataset file at %s", self.file_path)

        for index, example in enumerate(self.input_examples):
            line_input_items = example.text_a.strip().split(' ')
            line_label_items = self.label2id[example.label]

            input_token = ["[CLS]"] + line_input_items[:self.max_sen_len - 2] + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_token)
            self.input_ids.append(torch.LongTensor(input_ids))

            self.label_ids.append(line_label_items)

            if index < 3:
                logger.info("*** Example ***")
                logger.info("guid: {} ".format(example.guid))
                logger.info("tokens: %s" % " ".join([str(x) for x in input_token]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                if example.label is not None:
                    logger.info("labels: %s" % str(example.label))
                    logger.info("label_ids: %s" % str(line_label_items))

    def cache_feature(self):
        data = {
            "input_ids": self.input_ids,
            "label_ids": self.label_ids,
        }
        torch.save(data, self.cache_feature_file)
        logger.info(f"save features cache : {self.cache_feature_file}")

    def load_cache_feature(self):
        cached_data = torch.load(self.cache_feature_file)
        self.input_ids = cached_data["input_ids"]
        self.label_ids = cached_data["label_ids"]

        logger.info(f"load features cache from : {self.cache_feature_file} , total number : {len(self.input_ids)}")

    def get_collate_fn(self, task_name="cls", use_rdrop_loss=None):
        use_rdrop_loss = self.use_rdrop_loss if use_rdrop_loss is None else use_rdrop_loss

        collate_fn = collate_fn_cls_bert
        if use_rdrop_loss:
            collate_fn = collate_fn_cls_bert_rdrop

        # if self.dataset_name == Constants.DATASET_NAME_CLS_TNEWS:
        #     collate_fn = collate_fn_cls_bert_rdrop

        return collate_fn

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.label_ids[index],
        }


def collate_fn_cls_bert(features) -> Dict[str, Tensor]:
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_labels = [feature["labels"] for feature in features]

    batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)

    assert batch_input_ids.shape == batch_attention_mask.shape

    return {
        "input_ids": batch_input_ids,
        "labels": torch.LongTensor(batch_labels),
        "attention_mask": batch_attention_mask,
    }


def collate_fn_cls_bert_rdrop(features) -> Dict[str, Tensor]:
    """
    rdrop ： x -> [x,x]

    :param features:
    :return:
    """
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_labels = [feature["labels"] for feature in features]
    batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

    # 扩大一倍
    batch_input_ids = sum([[item, item] for item in batch_input_ids], [])
    batch_labels = sum([[item, item] for item in batch_labels], [])
    batch_attention_mask = sum([[item, item] for item in batch_attention_mask], [])

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)

    assert batch_input_ids.shape == batch_attention_mask.shape

    return {
        "input_ids": batch_input_ids,
        "labels": torch.LongTensor(batch_labels),
        "attention_mask": batch_attention_mask,
    }
