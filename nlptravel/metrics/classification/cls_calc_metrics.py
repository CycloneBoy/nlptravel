#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project nlptravel
# @File  : cls_calc_metrics.py
# @Author: sl
# @Date  : 2022/4/10 - 下午4:06

from copy import deepcopy

import numpy as np

import os
from collections import OrderedDict

from sklearn import metrics

from nlptravel.metrics.csc.csc_metric_dcn import CscMetricDcn
from nlptravel.utils.common_utils import CommonUtils
from nlptravel.utils.constant import Constants
from nlptravel.utils.file_utils import FileUtils
from nlptravel.utils.logger_utils import logger


class ClsCalcMetrics(object):

    @staticmethod
    def compute_metrics(eval_pred):
        """
        计算metric

        :param eval_pred:
        :return:
        """
        predictions, labels = eval_pred
        pred = np.argmax(predictions, axis=1)

        # {'micro', 'macro', 'samples','weighted', 'binary'} or None,
        # average = None
        average = 'binary'

        f1 = metrics.f1_score(labels, pred, average=average).tolist()
        # auc = metrics.roc_auc_score(labels, pred, average=average, multi_class="ovo")
        auc = metrics.roc_auc_score(labels, pred, multi_class="ovo")
        # auc = 0
        acc = metrics.accuracy_score(labels, pred)
        p = metrics.precision_score(labels, pred, average=average).tolist()
        r = metrics.recall_score(labels, pred, average=average).tolist()
        class_report = metrics.classification_report(labels, pred)
        class_cf = metrics.confusion_matrix(labels, pred)

        result = {
            "acc": acc,
            "auc_score": auc,
            "f1": f1,
            "p": p,
            "r": r,
        }

        logger.info(f"classification_report:\n {class_report}")
        logger.info(f"confusion_matrix:\n {class_cf}")
        logger.info(f"metric_result: {result}")

        return result

    @staticmethod
    def compute_metrics_for_token_classification(eval_pred):
        """
        计算metric    : token 级别分类

        :param eval_pred:
        :return:
        """
        all_predictions, all_labels = eval_pred

        # 槽位分类
        predictions_detect = all_predictions
        labels_detect = all_labels
        preds_detect = np.argmax(predictions_detect, axis=-1)

        labels_detect_src = []
        preds_detect_src = []

        labels_detect_list = []
        preds_detect_list = []

        seq_lens = CommonUtils.get_cls_intent_and_slot_sighan15_label()
        for index, pred in enumerate(labels_detect):
            seq_len = seq_lens[index]
            raw_preds = preds_detect[index][1:seq_len + 1]
            raw_labels = labels_detect[index][1:seq_len + 1]
            preds_detect_src.append(raw_preds)
            labels_detect_src.append(raw_labels)

            preds_detect_list.extend(raw_preds)
            labels_detect_list.extend(raw_labels)

        file_dir = f"{Constants.DATA_DIR}/test"

        average = 'binary'
        char_result = CommonUtils.compute_metrics_cls(labels=labels_detect_list, preds=preds_detect_list,
                                                      average=average)

        # 保存预测结果
        # cls_pred_result_sentence = [f"{pred}\t{label}" for pred, label in zip(preds, labels)]
        cls_char_pred_result_sentence = [f"{pred}\t{label}" for pred, label in
                                         zip(preds_detect_list, labels_detect_list)]

        pred_result_sentence = [" ".join([str(pred) for pred in preds]) for preds in preds_detect_src]
        label_result_sentence = [" ".join([str(pred) for pred in preds]) for preds in labels_detect_src]

        cls_predict_result_path = f"{file_dir}/cls_predict.txt"
        cls_char_predict_result_path = f"{file_dir}/cls_char_predict.txt"
        predict_result_path = f"{file_dir}/cls_detect_predict.txt"
        label_result_path = f"{file_dir}/cls_detect_label.txt"

        # FileUtils.save_to_text(cls_predict_result_path, "\n".join(cls_pred_result_sentence))
        FileUtils.save_to_text(cls_char_predict_result_path, "\n".join(cls_char_pred_result_sentence))
        FileUtils.save_to_text(predict_result_path, "\n".join(pred_result_sentence))
        FileUtils.save_to_text(label_result_path, "\n".join(label_result_sentence))

        # 计算评估指标
        eval_file_path = Constants.CSC_RLS_DATA_EVAL_15_CLS_LABEL_DIR_CSV
        dcn_results = CscMetricDcn.eval_predict(eval_file_path=eval_file_path, predict_result_path=predict_result_path,
                                                show_info=False)

        # dcn_results["metric_cls"] = result
        metric_soft_mask = deepcopy(dcn_results["metric_soft_mask_bert"])

        logger.info(f"eval_metric:")
        result = {}
        for key, val in metric_soft_mask.items():
            if str(key).endswith("counts"):
                continue
            result[key] = val
        result.update(char_result)

        name = "det"
        logger.info(
            f"det_sent: acc : {result['det_sent_acc']:.4f}  p : {result['det_sent_p']:.4f} r : {result['det_sent_r']:.4f} f1 : {result['det_sent_f1']:.4f} ")
        logger.info(
            f"cor_sent: acc : {result['cor_sent_acc']:.4f}  p : {result['cor_sent_p']:.4f} r : {result['cor_sent_r']:.4f} f1 : {result['cor_sent_f1']:.4f} ")
        logger.info(f"dcn_results: {dcn_results}")
        eval_metric_file = f"{file_dir}/cls_sighan15_metrics.json"
        FileUtils.dump_json(eval_metric_file, dcn_results)

        # return metric_soft_mask
        return result

    @staticmethod
    def compute_metrics_for_intent_and_slot(eval_pred):
        """
        计算metric    : 联合 意图和槽位

        :param eval_pred:
        :return:
        """
        all_predictions, all_labels = eval_pred

        # 意图分类
        predictions = all_predictions[0]
        labels = all_labels[0]
        preds = np.argmax(predictions, axis=1)

        # {'micro', 'macro', 'samples','weighted', 'binary'} or None,
        average = 'binary'
        result = CommonUtils.compute_metrics_cls(labels=labels, preds=preds, average=average)

        # 槽位分类
        predictions_detect = all_predictions[1]
        labels_detect = all_labels[1]
        preds_detect = np.argmax(predictions_detect, axis=-1)

        src_lens = all_labels[2]
        labels_detect_src = []
        preds_detect_src = []

        for index, seq_len in enumerate(src_lens):
            raw_preds = preds_detect[index][1:seq_len + 1]
            raw_labels = labels_detect[index][1:seq_len + 1]
            preds_detect_src.append(raw_preds)
            labels_detect_src.append(raw_labels)

        file_dir = f"{Constants.DATA_DIR}/test"

        # 保存预测结果
        cls_pred_result_sentence = [f"{pred}\t{label}" for pred, label in zip(preds, labels)]

        pred_result_sentence = [" ".join([str(pred) for pred in preds]) for preds in preds_detect_src]
        label_result_sentence = [" ".join([str(pred) for pred in preds]) for preds in labels_detect_src]

        cls_predict_result_path = f"{file_dir}/cls_predict.txt"
        predict_result_path = f"{file_dir}/cls_detect_predict.txt"
        label_result_path = f"{file_dir}/cls_detect_label.txt"

        FileUtils.save_to_text(cls_predict_result_path, "\n".join(cls_pred_result_sentence))
        FileUtils.save_to_text(predict_result_path, "\n".join(pred_result_sentence))
        FileUtils.save_to_text(label_result_path, "\n".join(label_result_sentence))

        # 计算评估指标
        eval_file_path = Constants.CSC_RLS_DATA_EVAL_15_CLS_LABEL_DIR_CSV
        dcn_results = CscMetricDcn.eval_predict(eval_file_path=eval_file_path, predict_result_path=predict_result_path,
                                                show_info=False)

        dcn_results["metric_cls"] = result
        metric_soft_mask = deepcopy(dcn_results["metric_soft_mask_bert"])
        result.update(metric_soft_mask)

        logger.info(f"dcn_results: {dcn_results}")
        eval_metric_file = f"{file_dir}/cls_sighan15_metrics.json"
        FileUtils.dump_json(eval_metric_file, dcn_results)

        return result
