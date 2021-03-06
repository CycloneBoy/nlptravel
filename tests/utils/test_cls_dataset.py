#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project nlptravel
# @File  : test_cls_dataset.py
# @Author: sl
# @Date  : 2022/4/10 - 下午3:01

import os.path
import shutil
import subprocess
import unittest
import wandb

from nlptravel.metrics.csc.csc_metric_dcn import CscMetricDcn
from nlptravel.utils.cmd_utils import CmdUtils
from nlptravel.utils.file_utils import FileUtils

from torch.utils.data import DataLoader
from transformers import ElectraTokenizer, BertTokenizer, AutoModelForSequenceClassification

from nlptravel.dataset.cls.cls_dataset_itent_slot import ClsIntentAndSlotDataset
from nlptravel.model.classification.cls_bert_model import JointBERT
from nlptravel.utils.constant import Constants
from nlptravel.utils.logger_utils import logger

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

        # tokenizer = BertTokenizer(vocab_file=Constants.BERT_VOCAB_FILE)
        # tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        eval_data_file = Constants.CSC_DATA_EVAL_15_DIR_CSV
        dataset = ClsIntentAndSlotDataset(file_path=eval_data_file, tokenizer=tokenizer, block_size=max_length)
        print(dataset[0])

        # dataset.save_to_file(Constants.CSC_DATA_EVAL_15_CLS_LABEL_DIR_CSV)

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

    def test_wandb_login(self):
        with open("/home/sl/wandb.token", mode='r') as f:
            token = str(f.readline()).replace("\n", "")
        wandb.login(key=token)

    def test_modify_constants_dir(self):
        CmdUtils.modify_constants_py_file()

    def test_eval_dcn(self):
        eval_file_path = Constants.CSC_RLS_DATA_EVAL_15_CLS_LABEL_DIR_CSV
        predict_result_path = "/home/sl/workspace/python/a2022/nlptravel/data/test/cls_detect_predict.txt"
        pred_path, orig_truth_path = CscMetricDcn.generate_sighan_format(eval_file_path,
                                                                         predict_result_path=predict_result_path,
                                                                         eval_ture_input=False,
                                                                         show_info=True, )

    def test_eval_cls(self):
        eval_file_path = Constants.CSC_RLS_DATA_EVAL_15_CLS_LABEL_DIR_CSV
        predict_result_path = "/home/sl/workspace/python/a2022/nlptravel/data/test/cls_detect_predict.txt"
        dcn_results = CscMetricDcn.eval_predict(eval_file_path=eval_file_path,
                                                    predict_result_path=predict_result_path,
                                                    show_info=True)

        logger.info(f"{dcn_results}")
