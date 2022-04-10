#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel 
# @File    ：train_cls.py
# @Author  ：sl
# @Date    ：2022/3/9 17:13
from copy import deepcopy

import numpy as np
from nlptravel.datasets.cls.cls_dataset_bert import ClsBertDataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AlbertTokenizer, AutoTokenizer, \
    PreTrainedTokenizer, AutoModelForTokenClassification

from transformers.training_args import default_logdir

import os
from collections import OrderedDict

import torch
from sklearn import metrics
from torch import nn
from transformers import set_seed, AutoConfig, CONFIG_MAPPING, BertTokenizer, ElectraTokenizer, ElectraConfig, \
    BertConfig

from nlptravel.datasets.cls.cls_dataset_itent_slot import ClsIntentAndSlotDataset
from nlptravel.entity.ner_common_entity import ModelArguments, DataTrainingArguments
from nlptravel.metrics.classification.cls_calc_metrics import ClsCalcMetrics
from nlptravel.model.classification.cls_bert_model import JointBERT
from nlptravel.trainer.cls_trainer import ClsTrainer

from nlptravel.utils.base_utils import ModelNameType, NlpPretrain
from nlptravel.utils.common_utils import CommonUtils
from nlptravel.utils.constant import Constants
from nlptravel.utils.file_utils import FileUtils
from nlptravel.utils.logger_utils import logger
from nlptravel.utils.time_utils import TimeUtils

"""
训练CLS 模型
"""

NUM_CLASS = 2
EVAL_BEGIN_TIME = TimeUtils.now_str_short()


def build_bert_cls_model(config, model_args: ModelArguments, data_args: DataTrainingArguments, ) -> nn.Module:
    """
    构造模型
    :param config:
    :param model_args:
    :param data_args:
    :return:
    """

    dataset_name = data_args.dataset_name
    num_labels = 2
    hidden_dropout_prob = 0.3 if model_args.use_rdrop_loss else 0.1
    model_name_or_path = model_args.model_name_or_path
    model_name = model_args.model_name

    if dataset_name == Constants.DATASET_NAME_CLS_TNEWS:
        num_labels = len(Constants.DATASET_NAME_CLS_TNEWS_LABELS)

    if model_name == ModelNameType.CLS_TEXT_CNN.desc:
        pass
    elif model_name == ModelNameType.CLS_JOINT_BERT.desc:
        model = JointBERT.from_pretrained(model_name_or_path, num_labels=num_labels,
                                          hidden_dropout_prob=hidden_dropout_prob,
                                          slot_loss_coef=model_args.loss_weight)
    elif model_name == ModelNameType.CLS_AUTOMODEL_FOR_TOKEN_CLASSIFICATION.desc:
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels,
                                                                hidden_dropout_prob=hidden_dropout_prob)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels,
                                                                   hidden_dropout_prob=hidden_dropout_prob)

    return model


def build_bert_dataset(training_args: TrainingArguments, tokenizer: PreTrainedTokenizer,
                       data_args: DataTrainingArguments,
                       model_args: ModelArguments):
    """
     构造dataset

    :param training_args:
    :param tokenizer:
    :param data_args:
    :param model_args:
    :return:
    """
    block_size = data_args.block_size

    dataset_name = data_args.dataset_name
    use_rdrop_loss = model_args.use_rdrop_loss
    model_name = model_args.model_name
    task_name = model_args.task_name

    data_file_list = FileUtils.get_dataset_path_cls(dataset_name=dataset_name)

    # 构建dataset
    train_dataset = None

    DATASET_CLASS = ClsBertDataset
    # 计算分类指标
    run_compute_metrics = ClsCalcMetrics.compute_metrics

    if model_name == ModelNameType.CLS_TEXT_CNN.desc:
        pass
    elif model_name == ModelNameType.CLS_JOINT_BERT.desc:
        DATASET_CLASS = ClsIntentAndSlotDataset
        #  标记label
        training_args.label_names = ["labels", "detect_labels", "src_lens"]
        run_compute_metrics = ClsCalcMetrics.compute_metrics_for_intent_and_slot
    elif model_name == ModelNameType.CLS_AUTOMODEL_FOR_TOKEN_CLASSIFICATION.desc:
        DATASET_CLASS = ClsIntentAndSlotDataset
        #  标记label
        # training_args.label_names = ["labels", "src_lens"]
        run_compute_metrics = ClsCalcMetrics.compute_metrics_for_token_classification
    if training_args.do_train:
        train_dataset = DATASET_CLASS(file_path=data_file_list["train"], tokenizer=tokenizer,
                                      block_size=block_size, dataset_name=dataset_name, use_rdrop_loss=use_rdrop_loss)

    eval_dataset = DATASET_CLASS(file_path=data_file_list["eval"], tokenizer=tokenizer,
                                 block_size=block_size, dataset_name=dataset_name, use_rdrop_loss=use_rdrop_loss)
    test_dataset = DATASET_CLASS(file_path=data_file_list["eval"], tokenizer=tokenizer,
                                 block_size=block_size, dataset_name=dataset_name, use_rdrop_loss=use_rdrop_loss)

    data_collator = eval_dataset.get_collate_fn(task_name=task_name)

    return train_dataset, eval_dataset, test_dataset, data_collator, run_compute_metrics


def main(model_name_or_path=None, run_log=None):
    """
    训练模型
    :return:
    """
    model_args, data_args, training_args = CommonUtils.parse_model_and_data_args(show_info=True)
    if model_name_or_path is not None:
        model_args.model_name_or_path = model_name_or_path
    if run_log is None:
        run_log = ["文本分类", ModelNameType.parse(model_args.model_name).desc]

    checkpoint_dir = CommonUtils.get_checkpoint_dir(model_args=model_args, data_args=data_args,
                                                    set_time=EVAL_BEGIN_TIME)
    training_args.output_dir = checkpoint_dir
    training_args.logging_dir = os.path.join(checkpoint_dir, default_logdir())

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path and model_args.model_name != ModelNameType.NER_BILSTM_CRF.desc:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        # tokenizer = BertTokenizer.from_pretrained(
        #     model_args.tokenizer_name, cache_dir=model_args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif model_args.model_name_or_path:
        # tokenizer = BertTokenizer.from_pretrained(
        # model_name_or_path = NlpPretrain.ROBERTA_CHINESE_WWM_EXT_PYTORCH.path
        model_name_or_path = model_args.model_name_or_path

        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = None

    # 加载预训练模型 "bert-base-chinese"
    if model_args.model_name_or_path:
        model = build_bert_cls_model(config, model_args=model_args, data_args=data_args)
    else:
        logger.warning("Training new model from scratch")
        raise ValueError("模型名称为空")

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    train_dataset, eval_dataset, test_dataset, data_collator, run_compute_metrics = build_bert_dataset(
        training_args=training_args,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args)

    # trainer
    trainer = ClsTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_args=data_args,
        model_args=model_args,
        run_log=run_log,
        compute_metrics=run_compute_metrics
    )

    # Training
    if training_args.do_train:
        train_begin_time = TimeUtils.now_str()
        logger.info(f"开始训练：{model_args.model_name} - {train_begin_time}")

        train_out = trainer.train()
        train_end_time = TimeUtils.now_str()
        logger.info(f"结束训练：{TimeUtils.calc_diff_time(train_begin_time, train_end_time)}")
        logger.info(f"训练指标： {train_out}")

        # 保存训练好的模型
        best_model_path = f"{checkpoint_dir}/best"
        trainer.save_model(best_model_path)
        logger.info(f"保存训练好的最好模型:{best_model_path}")

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(best_model_path)

    # Evaluation
    results = {}
    if training_args.do_eval:
        eval_begin_time = TimeUtils.now_str()
        logger.info(f"开始进行评估，{eval_begin_time}")
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()
        eval_end_time = TimeUtils.now_str()
        logger.info(f"结束评估,耗时：{TimeUtils.calc_diff_time(eval_begin_time, eval_end_time)} s")
        logger.info(f"评估指标： {eval_output}")

        results["eval"] = eval_output

    # Predict
    if training_args.do_predict:
        logger.info(f"开始进行预测")
        trainer.predict(test_dataset=test_dataset)
        logger.info(f"完成预测")

    return results


if __name__ == '__main__':
    pass
    main()
