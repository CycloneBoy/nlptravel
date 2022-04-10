#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project nlptravel
# @File  : test_cls_dataset.py
# @Author: sl
# @Date  : 2022/4/10 - 下午3:01

import os.path
import shutil
import unittest

from torch.utils.data import DataLoader
from transformers import ElectraTokenizer, BertTokenizer, AutoModelForSequenceClassification

from nlptravel.datasets.cls.cls_dataset_itent_slot import ClsIntentAndSlotDataset
from nlptravel.model.classification.cls_bert_model import JointBERT
from nlptravel.utils.constant import Constants

"""
测试模型和数据

"""


class DatasetTest(unittest.TestCase):

    def test_load_sighan_test(self):
        batch_size = 4
        max_length = 128
        num_labels = 2

        # model_name_or_path = "voidful/albert_chinese_tiny"
        # model_name_or_path = "hfl/chinese-electra-180g-small-discriminator"
        model_name_or_path = Constants.ELECTRA_SAMLL_DISCRIMINATOR
        # model_name_or_path = NlpPretrain.ROBERTA_CHINESE_WWM_EXT_PYTORCH.path

        tokenizer = BertTokenizer(vocab_file=Constants.BERT_VOCAB_FILE)
        # tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)

        eval_data_file = Constants.CSC_DATA_EVAL_15_DIR_CSV
        dataset = ClsIntentAndSlotDataset(file_path=eval_data_file, tokenizer=tokenizer, block_size=max_length)
        print(dataset[0])

        # datasets.save_to_file(Constants.CSC_DATA_EVAL_15_CLS_LABEL_DIR_CSV)

        collate_fn = dataset.get_collate_fn()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        print(f"total:{len(dataloader)}")
        print(batch)

        # model = JointBERT.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)

        print(model)
        # # model.from_pretrained(plome_torch_path, config=config)
        # # model.load_state_dict(model_dict)

        for name, parameters in model.named_parameters():  # 打印出每一层的参数的大小
            print(name, ':', parameters.size())

        output = model(**batch)
        print(output)
