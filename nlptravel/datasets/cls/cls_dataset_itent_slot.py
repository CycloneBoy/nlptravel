#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel 
# @File    ：cls_dataset_itent_slot.py
# @Author  ：sl
# @Date    ：2022/3/29 14:40

import os
from dataclasses import dataclass

from typing import List, Dict, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import random
import pandas as pd
from tqdm import tqdm

from transformers import PreTrainedTokenizer

from nlptravel.utils.base_utils import DataFileType
from nlptravel.utils.common_utils import CommonUtils
from nlptravel.utils.constant import Constants
from nlptravel.utils.file_utils import FileUtils
from nlptravel.utils.logger_utils import logger

"""
intent 和slot 联合dataset 

"""


class ClsIntentAndSlotDataset(Dataset):

    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 block_size: int = 180,
                 need_cache=False, input_examples=None, dataset_name=None,
                 use_rdrop_loss=False, *args, **kwargs):

        # print(file_path, os.path.isfile(file_path))
        assert os.path.isfile(file_path)

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
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
        self.detect_label_ids = []
        self.input_lens = []
        self.count = 0

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
            logger.info("Creating features from datasets file at %s", self.file_path)

        # csc 两列 转
        for index, example in enumerate(tqdm(self.input_examples)):
            line_input_items = example.text_a.strip().split()
            line_label_items = example.label.strip().split()

            if len(line_input_items) != len(line_label_items):
                logger.info(f"长度不一致：{example}")

            input_token = ["[CLS]"] + line_input_items[:self.block_size - 2] + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_token)
            label_token = ["[CLS]"] + line_label_items[:self.block_size - 2] + ["[SEP]"]
            label_ids = self.tokenizer.convert_tokens_to_ids(label_token)

            detect_label_ids = CommonUtils.build_detect_labels(input_ids, label_ids)
            seq_len = len(input_ids) - 2

            error_num = sum(detect_label_ids)

            # 完全正确句子
            cls_label = self.label2id["0"]
            self.input_ids.append(torch.LongTensor(label_ids))
            self.label_ids.append(cls_label)

            # 全0
            detect_ids = [0 for _ in range(0, len(label_ids))]
            self.detect_label_ids.append(torch.LongTensor(detect_ids))
            self.input_lens.append(seq_len)

            # 句子有错误
            if error_num > 0:
                cls_label = self.label2id["1"]
                self.input_ids.append(torch.LongTensor(input_ids))
                self.label_ids.append(cls_label)

                self.detect_label_ids.append(torch.LongTensor(detect_label_ids))
                self.input_lens.append(seq_len)

            if index < 3:
                logger.info("*** Example ***")
                logger.info("guid: {} ".format(example.guid))
                logger.info("tokens: %s" % " ".join([str(x) for x in input_token]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                if example.label is not None:
                    logger.info("labels: %s" % str(example.label))
                    logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                    logger.info("detect_label_ids: %s " % " ".join([str(x) for x in detect_label_ids]))

        self.count = len(self.input_ids)
        logger.info(f"dataset加载完毕：{self.count} 原始：{len(self.input_examples)}")
        self.input_examples = []

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

    def save_to_file(self, file_name):

        results = []
        for index, input_id in enumerate(self.input_ids):
            input_lens = self.input_lens[index]

            input_id = [str(token) for token in input_id.numpy().tolist()][1:input_lens + 1]
            labels = self.label_ids[index]
            detect_labels = [str(token) for token in self.detect_label_ids[index].numpy().tolist()][1:input_lens + 1]

            res = [" ".join(input_id), " ".join(detect_labels), str(labels), str(input_lens)]
            results.append("\t".join(res))

        logger.info(f"保存数据集到：{file_name} - {len(results)}")
        FileUtils.save_to_text(filename=file_name, content="\n".join(results))

    def get_collate_fn(self, task_name="intent_and_slot"):
        collate_fn = DataCollatorClsIntentAndSlot()
        if task_name != "intent_and_slot":
            collate_fn = DataCollatorClsIntentAndSlot(task_name=task_name)
        return collate_fn

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.label_ids[index],
            "detect_labels": self.detect_label_ids[index],
            "input_lens": self.input_lens[index],
        }


def collate_fn_intent_and_slot(features) -> Dict[str, Tensor]:
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_detect_labels = [feature["detect_labels"] for feature in features]

    batch_labels = [feature["labels"] for feature in features]
    input_lens = [feature["input_lens"] for feature in features]

    batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

    # loss mask
    batch_attention_mask_label = []
    for feature in features:
        seq_len = feature["input_lens"]
        attention_mask = torch.concat(
            [torch.zeros(1), torch.ones_like(feature["input_ids"])[1:seq_len + 1], torch.zeros(1)], dim=0)
        batch_attention_mask_label.append(attention_mask)

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_detect_labels = pad_sequence(batch_detect_labels, batch_first=True, padding_value=0)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    batch_attention_mask_label = pad_sequence(batch_attention_mask_label, batch_first=True, padding_value=0)

    assert batch_input_ids.shape == batch_detect_labels.shape
    assert batch_input_ids.shape == batch_attention_mask.shape
    assert batch_input_ids.shape == batch_attention_mask_label.shape

    return {
        "input_ids": batch_input_ids,
        "labels": torch.LongTensor(batch_labels),
        "detect_labels": batch_detect_labels,
        "attention_mask": batch_attention_mask,
        "loss_mask": batch_attention_mask_label,
        "src_lens": torch.LongTensor(input_lens),
    }


@dataclass
class DataCollatorClsIntentAndSlot:
    task_name: str = "intent_and_slot"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [feature["input_ids"] for feature in features]
        batch_detect_labels = [feature["detect_labels"] for feature in features]

        batch_labels = [feature["labels"] for feature in features]
        input_lens = [feature["input_lens"] for feature in features]

        batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

        # loss mask
        batch_attention_mask_label = []
        for feature in features:
            seq_len = feature["input_lens"]
            attention_mask = torch.concat(
                [torch.zeros(1), torch.ones_like(feature["input_ids"])[1:seq_len + 1], torch.zeros(1)], dim=0)
            batch_attention_mask_label.append(attention_mask)

        # padding
        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
        batch_detect_labels = pad_sequence(batch_detect_labels, batch_first=True, padding_value=0)
        batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        batch_attention_mask_label = pad_sequence(batch_attention_mask_label, batch_first=True, padding_value=0)

        assert batch_input_ids.shape == batch_detect_labels.shape
        assert batch_input_ids.shape == batch_attention_mask.shape
        assert batch_input_ids.shape == batch_attention_mask_label.shape

        data = {
            "input_ids": batch_input_ids,
            "labels": torch.LongTensor(batch_labels),
            "attention_mask": batch_attention_mask,
        }

        if self.task_name != "intent_and_slot":
            data["labels"] = batch_detect_labels

        return data
