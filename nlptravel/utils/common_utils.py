#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project nlptravel
# @File  : common_utils.py
# @Author: sl
# @Date  : 2022/4/10 - 下午2:53
from typing import Tuple

from sklearn import metrics
from transformers import TrainingArguments, HfArgumentParser

from nlptravel.entity.ner_common_entity import ModelArguments, DataTrainingArguments
from nlptravel.utils.base_utils import BaseUtil
from nlptravel.utils.constant import Constants
from nlptravel.utils.file_utils import FileUtils
from nlptravel.utils.logger_utils import logger
from nlptravel.utils.time_utils import TimeUtils

"""
通用工具类
"""


class CommonUtils(BaseUtil):
    """
    通用工具类
    """

    def init(self):
        pass

    @staticmethod
    def parse_model_and_data_args(show_info=True) -> Tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
        """
        解析命令行参数 model_args, data_args
        :return:
        """
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        if show_info:
            logger.info(model_args)
            logger.info(data_args)
        # logger.info(training_args)

        # 改变eval sighan 时间
        if str(model_args.task_name).lower() == "csc" and data_args.eval_data_file == Constants.CSC_RLS_DATA_EVAL_15_DIR:
            testset_year = data_args.testset_year
            if testset_year == 14:
                data_args.eval_data_file = Constants.CSC_RLS_DATA_EVAL_14_DIR
                data_args.eval_label_file = Constants.CSC_RLS_DATA_EVAL_LABEL_14_DIR

                data_args.test_data_file = Constants.CSC_RLS_DATA_EVAL_14_DIR
            elif testset_year == 13:
                data_args.eval_data_file = Constants.CSC_RLS_DATA_EVAL_13_DIR
                data_args.eval_label_file = Constants.CSC_RLS_DATA_EVAL_LABEL_13_DIR

                data_args.test_data_file = Constants.CSC_RLS_DATA_EVAL_13_DIR

            logger.info(f"改变eval sighan数据集：{testset_year}")

        return model_args, data_args, training_args

    @staticmethod
    def get_checkpoint_dir(model_args: ModelArguments, data_args: DataTrainingArguments, set_time=None):
        """
        获取checkpoint dir
        :param model_args:
        :param data_args:
        :param set_time:
        :return:
        """
        if set_time is None:
            set_time = TimeUtils.now_str_short()
        checkpoint_dir = f"{data_args.output_file}/models/{model_args.model_name}_{model_args.task_name}/{set_time}"
        return checkpoint_dir

    @staticmethod
    def build_detect_labels(input_ids, label_ids):
        """
        添加检测的label
        :param input_ids:
        :param label_ids:
        :return:
        """
        detect_label_ids = []
        for index, (src, label) in enumerate(zip(input_ids, label_ids)):
            if src != label:
                detect_label_ids.append(1)
            else:
                detect_label_ids.append(0)
        return detect_label_ids

    @staticmethod
    def get_cls_intent_and_slot_sighan15_label(item_index=3):
        """
            读取分类的标签 长度
        872 1962 8013 2769 3221 2476 4263 3152 511      0 0 0 0 0 0 0 0 0       0       9
        :return:
        """
        eval_file_path = Constants.CSC_DATA_EVAL_15_CLS_LABEL_DIR_CSV
        raw_list = FileUtils.read_to_text_list(eval_file_path)
        results = []
        for index, line in enumerate(raw_list):
            splits = str(line).strip().split("\t")
            item = int(splits[item_index])
            results.append(item)

        return results

    @staticmethod
    def compute_metrics_cls(labels, preds, average='binary'):
        """
        计算 分类指标
        :param labels:
        :param preds:
        :param average:
        :return:
        """
        average_auc = average
        if average == "binary":
            average_auc = None
        f1 = metrics.f1_score(labels, preds, average=average).tolist()
        auc = metrics.roc_auc_score(labels, preds, average=average_auc, multi_class="ovo")
        acc = metrics.accuracy_score(labels, preds)
        p = metrics.precision_score(labels, preds, average=average).tolist()
        r = metrics.recall_score(labels, preds, average=average).tolist()

        class_report = metrics.classification_report(labels, preds)
        class_cf = metrics.confusion_matrix(labels, preds)

        result = {
            "acc": acc,
            "auc_score": auc,
            "f1": f1,
            "p": p,
            "r": r,
        }

        logger.info(f"classification metric: ")
        for k, v in result.items():
            logger.info(f"{k} : {v:.4f}")

        logger.info(f"classification_report:\n{class_report}")
        logger.info(f"confusion_matrix:\n{class_cf}")
        logger.info(f"metric_result: {result}")

        return result

    @staticmethod
    def extract_sighan15_format(pred_list, target_list, input_index=None):
        """
        提取 sighan15 格式的预测数据
        :param pred_list:
        :param target_list:
        :param input_index:
        :return: 输出 predict 结果
        """
        result = []
        for index, (preds, targets) in enumerate(zip(pred_list, target_list)):
            target = str(targets).split()
            pred = str(preds).split()[:len(target)]

            if input_index is not None:
                tid = input_index[index]
            else:
                tid = index + 1
            one_result = CommonUtils.extract_sighan15_format_one(pred, target, tid=tid, )
            result.append(one_result)
        return result

    @staticmethod
    def extract_sighan15_format_one(pred, target, tid=0):
        """
         提取 sighan15 格式的预测数据 一句话
        :param pred: 保存 pred
        :param target:
        :param tid:
        :return:
        """
        output_list = [str(tid)]
        for i, (pt, at) in enumerate(zip(pred[:], target[:])):
            if at == "[SEP]" or at == '[PAD]':
                break
            # Post preprocess with unsupervised methods,
            # because unsup BERT always predict punchuation at 1st pos
            if i == 0:
                if pt == "。" or pt == "，":
                    continue
            if pt.startswith("##"):
                pt = pt.lstrip("##")
            if at.startswith("##"):
                at = at.lstrip("##")
            if pt != at:
                output_list.append(str(i + 1))
                output_list.append(pt)

        if len(output_list) == 1:
            output_list.append("0")

        return ", ".join(output_list)

