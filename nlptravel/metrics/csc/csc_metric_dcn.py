#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel
# @File    ：csc_metric_dcn.py
# @Author  ：sl
# @Date    ：2022/2/28 15:05


import pandas as pd

from nlptravel.metrics.csc.metric_core import metric_file
from nlptravel.utils.common_utils import CommonUtils
from nlptravel.utils.constant import Constants
from nlptravel.utils.file_utils import FileUtils
from nlptravel.utils.logger_utils import logger
from nlptravel.utils.time_utils import TimeUtils

"""
csc metric dcn
"""


class CscMetricDcn(object):

    @staticmethod
    def eval_predict(eval_file_path, predict_result_path, eval_ture_input=False, show_info=False):
        """
        预测结果

        :param eval_file_path:  原始的文件
        :param predict_result_path: 预测的结果，带序号
        :param eval_ture_input:
        :param show_info:
        :return:
        """
        input_examples = FileUtils.read_data_with_file_end(file_name=eval_file_path, show_info=show_info,
                                                           eval_ture_input=eval_ture_input)

        df = pd.read_csv(predict_result_path, sep="\t", names=['text'])

        # (original_text, correct_text, predict_text,)
        results = []

        results_pad = []
        length_not_same = 0
        for index, example in enumerate(input_examples):
            predict = df.iloc[index, 0]
            example.text_b = predict
            if eval_ture_input:
                example.text_a = example.label

            if isinstance(example.text_a, str):
                src = str(example.text_a).split()
            else:
                src = example.text_a

            if isinstance(example.label, str):
                target = str(example.label).split()
            else:
                target = example.label

            if isinstance(predict, str):
                pred = str(predict).split()
                if len(pred) < 3:
                    pred = list(predict)
                pred = pred[:len(target)]
            else:
                pred = predict

            if len(src) == len(target) and len(src) == len(pred):
                results.append((src, target, pred))
            else:
                length_not_same += 1
                if length_not_same < 2:
                    logger.info(f"数据长度不一致：index :{index} ")
                    logger.info(f"src: {example.text_a}")
                    logger.info(f"tag: {example.label}")
                    logger.info(f"prd: {predict}")
                    logger.info("")

        if show_info:
            logger.info(
                f"数据长度不一致数量: {length_not_same} / {len(input_examples)} - {length_not_same / len(input_examples)}")
            logger.info("--------------------- EVAL SoftMaskBert metric------------------------------")

        eval_logger = logger if show_info else None
        metric_soft_mask_bert = CscMetricDcn.compute_corrector_prf_faspell(results, logger=eval_logger, strict=True)

        if show_info:
            logger.info("--------------------- EVAL Dcn metric------------------------------")
        pred_path, orig_truth_path = CscMetricDcn.generate_sighan_format(eval_file_path,
                                                                         predict_result_path=predict_result_path,
                                                                         eval_ture_input=eval_ture_input,
                                                                         show_info=show_info, )

        metric_dcn = CscMetricDcn.eval_spell(truth_path=orig_truth_path, pred_path=pred_path, with_error=False,
                                             show_info=show_info)

        if show_info:
            logger.info(("metric_dcn : {}".format(metric_dcn)))

        metric_csc = {"metric_soft_mask_bert": metric_soft_mask_bert, "metric_dcn": metric_dcn}
        return metric_csc

    @staticmethod
    def compute_corrector_prf_faspell(results, logger=None, strict=True):
        """
        All-in-one measure function.
        based on FASpell's measure script.
        :param results: a list of (wrong, correct, predict, ...)
        both token_ids or characters are fine for the script.
        :param logger: take which logger to print logs.
        :param strict: a more strict evaluation mode (all-char-detected/corrected)
        References:
            sentence-level PRF: https://github.com/iqiyi/
            FASPell/blob/master/faspell.py
        """

        corrected_char, wrong_char = 0, 0
        corrected_sent, wrong_sent = 0, 0
        true_corrected_char = 0
        true_corrected_sent = 0
        true_detected_char = 0
        true_detected_sent = 0
        accurate_detected_sent = 0
        accurate_corrected_sent = 0
        all_sent = 0

        for item in results:
            # wrong, correct, predict, d_tgt, d_predict = item
            wrong, correct, predict = item[:3]

            all_sent += 1
            wrong_num = 0
            corrected_num = 0
            original_wrong_num = 0
            true_detected_char_in_sentence = 0

            for c, w, p in zip(correct, wrong, predict):
                if c != p:
                    wrong_num += 1
                if w != p:
                    corrected_num += 1
                    if c == p:
                        true_corrected_char += 1
                    if w != c:
                        true_detected_char += 1
                        true_detected_char_in_sentence += 1
                if c != w:
                    original_wrong_num += 1

            corrected_char += corrected_num
            wrong_char += original_wrong_num
            if original_wrong_num != 0:
                wrong_sent += 1
            if corrected_num != 0 and wrong_num == 0:
                true_corrected_sent += 1

            if corrected_num != 0:
                corrected_sent += 1

            if strict:  # find out all faulty wordings' potisions
                true_detected_flag = (true_detected_char_in_sentence == original_wrong_num
                                      and original_wrong_num != 0
                                      and corrected_num == true_detected_char_in_sentence)
            else:  # think it has faulty wordings
                true_detected_flag = (corrected_num != 0 and original_wrong_num != 0)

            # if corrected_num != 0 and original_wrong_num != 0:
            if true_detected_flag:
                true_detected_sent += 1
            if correct == predict:
                accurate_corrected_sent += 1
            if correct == predict or true_detected_flag:
                accurate_detected_sent += 1

        counts = {  # TP, FP, TN for each level
            'det_char_counts': [true_detected_char,
                                corrected_char - true_detected_char,
                                wrong_char - true_detected_char],
            'cor_char_counts': [true_corrected_char,
                                corrected_char - true_corrected_char,
                                wrong_char - true_corrected_char],
            'det_sent_counts': [true_detected_sent,
                                corrected_sent - true_detected_sent,
                                wrong_sent - true_detected_sent],
            'cor_sent_counts': [true_corrected_sent,
                                corrected_sent - true_corrected_sent,
                                wrong_sent - true_corrected_sent],
            'det_sent_acc': accurate_detected_sent / all_sent,
            'cor_sent_acc': accurate_corrected_sent / all_sent,
            'all_sent_count': all_sent,
        }

        details = {}
        for phase in ['det_char', 'cor_char', 'det_sent', 'cor_sent']:
            dic = CscMetricDcn.report_prf(
                *counts[f'{phase}_counts'],
                phase=phase, logger=logger,
                return_dict=True)
            details.update(dic)
        details.update(counts)
        return details

    @staticmethod
    def report_prf(tp, fp, fn, phase, logger=None, return_dict=False):
        # For the detection Precision, Recall and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        if phase and logger:
            logger.info(f"The {phase} result is: "
                        f"{precision:.4f}/{recall:.4f}/{f1_score:.4f} -->\n"
                        # f"precision={precision:.6f}, recall={recall:.6f} and F1={f1_score:.6f}\n"
                        f"support: TP={tp}, FP={fp}, FN={fn}")
        if return_dict:
            ret_dict = {
                f'{phase}_p': precision,
                f'{phase}_r': recall,
                f'{phase}_f1': f1_score}
            return ret_dict
        return precision, recall, f1_score

    @staticmethod
    def generate_sighan_format(eval_file_path, predict_result_path, eval_ture_input=False, show_info=False, ):
        """
        生成sighan15 测试格式数据
        :return:
        """
        input_examples = FileUtils.read_data_with_file_end(file_name=eval_file_path, show_info=show_info,
                                                           eval_ture_input=eval_ture_input)

        df = pd.read_csv(predict_result_path, sep="\t", names=['text'])

        src_list = []
        target_list = []
        pred_list = []
        id_list = []
        for index, example in enumerate(input_examples):
            predict = df.iloc[index, 0]

            if isinstance(predict, str):
                pred = str(predict).split()
                if len(pred) < 3:
                    pred = list(predict)
                predict = pred[:len(example.label)]

            example.text_b = " ".join(predict)

            if eval_ture_input:
                example.text_a = example.label

            id_list.append(example.guid)
            src_list.append(example.text_a)
            target_list.append(example.label)
            pred_list.append(example.text_b)

        predict_result = CommonUtils.extract_sighan15_format(pred_list, src_list, input_index=id_list)
        target_result = CommonUtils.extract_sighan15_format(target_list, src_list, input_index=id_list)

        pred_path = FileUtils.get_file_path_csc_rls_output(file_path=eval_file_path, sep_name="predict_result_sighan")
        FileUtils.save_to_text(pred_path, "\n".join(predict_result))

        if show_info:
            logger.info(f"保存SIGHEN15格式预测结果：{pred_path} - {len(predict_result)} ")

        orig_truth_path = FileUtils.get_file_path_csc_rls_output(file_path=eval_file_path,
                                                                 sep_name="original_truth_sighan")
        FileUtils.save_to_text(orig_truth_path, "\n".join(target_result))
        if show_info:
            logger.info(f"保存SIGHEN15格式真实结果：{orig_truth_path} - {len(target_result)} ")

            msg = "********* Evaluation Test format sighan15 Sentence-level ************"
            print(msg)
            logger.info(msg)
            print(f"{len(predict_result)} - {len(target_result)} ")

        return pred_path, orig_truth_path

    @staticmethod
    def eval_spell(truth_path, pred_path, with_error=True, show_info=False):
        metric_dict = {}
        # Compute F1-score
        detect_TP, detect_FP, detect_FN = 0, 0, 0
        correct_TP, correct_FP, correct_FN = 0, 0, 0
        detect_sent_TP, sent_P, sent_N, correct_sent_TP = 0, 0, 0, 0
        dc_TP, dc_FP, dc_FN = 0, 0, 0
        for idx, (pred, actual) in enumerate(zip(open(pred_path, "r"),
                                                 open(truth_path, "r") if with_error else
                                                 open(truth_path, "r"))):
            pred_tokens = pred.strip().split(" ")
            actual_tokens = actual.strip().split(" ")
            # assert pred_tokens[0] == actual_tokens[0]
            pred_tokens = pred_tokens[1:]
            actual_tokens = actual_tokens[1:]
            detect_actual_tokens = [int(actual_token.strip(",")) \
                                    for i, actual_token in enumerate(actual_tokens) if i % 2 == 0]
            correct_actual_tokens = [actual_token.strip(",") \
                                     for i, actual_token in enumerate(actual_tokens) if i % 2 == 1]
            detect_pred_tokens = [int(pred_token.strip(",")) \
                                  for i, pred_token in enumerate(pred_tokens) if i % 2 == 0]
            _correct_pred_tokens = [pred_token.strip(",") \
                                    for i, pred_token in enumerate(pred_tokens) if i % 2 == 1]

            # Postpreprocess for ACL2019 csc paper which only deal with last detect positions in test data.
            # If we wanna follow the ACL2019 csc paper, we should take the detect_pred_tokens to:

            max_detect_pred_tokens = detect_pred_tokens

            correct_pred_zip = zip(detect_pred_tokens, _correct_pred_tokens)
            correct_actual_zip = zip(detect_actual_tokens, correct_actual_tokens)

            if detect_pred_tokens[0] != 0:
                sent_P += 1
                if sorted(correct_pred_zip) == sorted(correct_actual_zip):
                    correct_sent_TP += 1
            if detect_actual_tokens[0] != 0:
                if sorted(detect_actual_tokens) == sorted(detect_pred_tokens):
                    detect_sent_TP += 1
                sent_N += 1

            if detect_actual_tokens[0] != 0:
                detect_TP += len(set(max_detect_pred_tokens) & set(detect_actual_tokens))
                detect_FN += len(set(detect_actual_tokens) - set(max_detect_pred_tokens))
            detect_FP += len(set(max_detect_pred_tokens) - set(detect_actual_tokens))

            correct_pred_tokens = []
            # Only check the correct postion's tokens
            for dpt, cpt in zip(detect_pred_tokens, _correct_pred_tokens):
                if dpt in detect_actual_tokens:
                    correct_pred_tokens.append((dpt, cpt))

            correct_TP += len(set(correct_pred_tokens) & set(zip(detect_actual_tokens, correct_actual_tokens)))
            correct_FP += len(set(correct_pred_tokens) - set(zip(detect_actual_tokens, correct_actual_tokens)))
            correct_FN += len(set(zip(detect_actual_tokens, correct_actual_tokens)) - set(correct_pred_tokens))

            # Caluate the correction level which depend on predictive detection of BERT
            dc_pred_tokens = zip(detect_pred_tokens, _correct_pred_tokens)
            dc_actual_tokens = zip(detect_actual_tokens, correct_actual_tokens)
            dc_TP += len(set(dc_pred_tokens) & set(dc_actual_tokens))
            dc_FP += len(set(dc_pred_tokens) - set(dc_actual_tokens))
            dc_FN += len(set(dc_actual_tokens) - set(dc_pred_tokens))

        detect_precision = detect_TP * 1.0 / (detect_TP + detect_FP) if detect_TP + detect_FP > 0 else 0
        detect_recall = detect_TP * 1.0 / (detect_TP + detect_FN) if detect_TP + detect_FN > 0 else 0
        detect_F1 = 2. * detect_precision * detect_recall / (
                (detect_precision + detect_recall) + 1e-8) if detect_precision + detect_recall > 0 else 0

        correct_precision = correct_TP * 1.0 / (correct_TP + correct_FP + 1e-8) if correct_TP + correct_FP > 0 else 0
        correct_recall = correct_TP * 1.0 / (correct_TP + correct_FN + 1e-8) if correct_TP + correct_FN > 0 else 0
        correct_F1 = 2. * correct_precision * correct_recall / (
                (correct_precision + correct_recall) + 1e-8) if correct_precision + correct_recall > 0 else 0

        dc_precision = dc_TP * 1.0 / (dc_TP + dc_FP + 1e-8) if dc_TP + dc_FP > 0 else 0
        dc_recall = dc_TP * 1.0 / (dc_TP + dc_FN + 1e-8) if dc_TP + dc_FN > 0 else 0
        dc_F1 = 2. * dc_precision * dc_recall / (dc_precision + dc_recall + 1e-8) if dc_precision + dc_recall > 0 else 0
        if show_info:
            # Token-level metrics
            print("detect_precision=%f, detect_recall=%f, detect_Fscore=%f" % (
                detect_precision, detect_recall, detect_F1))
            print("correct_precision=%f, correct_recall=%f, correct_Fscore=%f" % (
                correct_precision, correct_recall, correct_F1))
            print("dc_joint_precision=%f, dc_joint_recall=%f, dc_joint_Fscore=%f" % (dc_precision, dc_recall, dc_F1))

        metric_dict["token_level"] = {
            "detect": {
                "precision": detect_precision,
                "recall": detect_recall,
                "f1": detect_F1,
            },
            "correct": {
                "precision": correct_precision,
                "recall": correct_recall,
                "f1": correct_F1,
            },
            "joint": {
                "precision": dc_precision,
                "recall": dc_recall,
                "f1": dc_F1,
            },
        }

        detect_sent_precision = detect_sent_TP * 1.0 / (sent_P) if sent_P > 0 else 0
        detect_sent_recall = detect_sent_TP * 1.0 / (sent_N) if sent_N > 0 else 0
        detect_sent_F1 = 2. * detect_sent_precision * detect_sent_recall / (
                (
                        detect_sent_precision + detect_sent_recall) + 1e-8) if detect_sent_precision + detect_sent_recall > 0 else 0

        correct_sent_precision = correct_sent_TP * 1.0 / (sent_P) if sent_P > 0 else 0
        correct_sent_recall = correct_sent_TP * 1.0 / (sent_N) if sent_N > 0 else 0
        correct_sent_F1 = 2. * correct_sent_precision * correct_sent_recall / (
                (
                        correct_sent_precision + correct_sent_recall) + 1e-8) if correct_sent_precision + correct_sent_recall > 0 else 0

        if show_info:
            # Sentence-level metrics
            print("detect_sent_precision=%f, detect_sent_recall=%f, detect_Fscore=%f" % (
                detect_sent_precision, detect_sent_recall, detect_sent_F1))
            print("correct_sent_precision=%f, correct_sent_recall=%f, correct_Fscore=%f" % (
                correct_sent_precision, correct_sent_recall, correct_sent_F1))

        metric_dict["sentence_level"] = {
            "detect": {
                "precision": detect_sent_precision,
                "recall": detect_sent_recall,
                "f1": detect_sent_F1,
                "support": {
                    "TP": detect_sent_TP,
                    "FP": sent_P - detect_sent_TP,
                    "FN": sent_N - detect_sent_TP,
                }
            },
            "correct": {
                "precision": correct_sent_precision,
                "recall": correct_sent_recall,
                "f1": correct_sent_F1,
                "support": {
                    "TP": correct_sent_TP,
                    "FP": sent_P - correct_sent_TP,
                    "FN": sent_N - correct_sent_TP,
                }
            },
        }

        if show_info:
            print("metric_dict : {}".format(metric_dict))
        return metric_dict

    @staticmethod
    def eval_csc_metric(pred_labels_path, true_labels_path, pred_raw_path, eval_file_path, show_info=False):
        """
        评估 csc 的 metric
        :param pred_labels_path:
        :param true_labels_path:
        :param pred_raw_path:
        :param eval_file_path:
        :param show_info:
        :return:
        """

        rls_results = metric_file(pred_path=pred_labels_path, targ_path=true_labels_path, use_decimal=True)

        if show_info:
            logger.info("--------------------- EVAL Rls metric------------------------------")
            for k, v in rls_results.items():
                print(f'{k}: {v}')
            logger.info(rls_results)

        dcn_results = CscMetricDcn.eval_predict(eval_file_path=eval_file_path, predict_result_path=pred_raw_path,
                                                show_info=show_info)
        if show_info:
            logger.info(dcn_results)

        dcn_results['metric_rls'] = rls_results

        return dcn_results

    @staticmethod
    def generate_sighan_format_label_file(eval_file_path, eval_ture_input=False, show_info=True):
        """
        提取 输入数据集的原始label  生成sighan15 测试格式数据
        :param eval_file_path:
        :param eval_ture_input:
        :param show_info:
        :return:
        """
        input_examples = FileUtils.read_csc_data_from_pickle(file_name=eval_file_path, return_str=True,
                                                             show_info=show_info, eval_ture_input=eval_ture_input)
        src_list = []
        target_list = []
        guid_list = []
        for index, example in enumerate(input_examples):
            if eval_ture_input:
                example.text_a = example.label

            src_list.append(example.text_a)
            target_list.append(example.label)
            guid_list.append(example.guid)

        target_result = CommonUtils.extract_sighan15_format(target_list, src_list, input_index=guid_list)

        orig_truth_path = FileUtils.get_file_path_csc_rls_output(file_path=eval_file_path, sep_name="original_truth")
        FileUtils.save_to_text(orig_truth_path, "\n".join(target_result))
        if show_info:
            logger.info(f"保存SIGHEN15格式真实结果：{orig_truth_path} - {len(target_result)} ")


def demo_metric():
    """
    评估指标测试
/home/mqq/shenglei/ner/financial_ner/outputs/models/bert-pho2-res-arch3_csc/20220331_143900/eval/test.sighan15_original_truth_sighan_2022-03-31_rls.txt
/home/mqq/shenglei/ner/financial_ner/outputs/models/bert-pho2-res-arch3_csc/20220331_143900/test.sighan15_predict_result_none_post_2022-03-31_rls.txt

    :return:
    """
    eval_time = "20220310_163625"
    eval_time = "20220310_170156"

    # file_path = f"/home/mqq/shenglei/ner/financial_ner/outputs/models/SpellBertPho2ResArch3_csc/{eval_time}"
    file_path = f"/home/mqq/shenglei/ner/financial_ner/outputs/models/bert-pho2-res-arch3_csc/20220331_143900"
    pred_labels_path = f"{file_path}/eval/labels.txt"
    # pred_raw_path = f"{file_path}/eval/preds.txt"
    pred_raw_path = f"{file_path}/test.sighan15_predict_result_2022-03-31_rls.txt"
    pred_raw_path = f"{file_path}/test.sighan15_predict_result_none_post_2022-03-31_rls.txt"

    # true_labels_path = f"/home/mqq/shenglei/data/nlp/csc/ReaLiSe/data/test.sighan15.lbl.tsv"
    true_labels_path = f"{file_path}/eval/test.sighan15_original_truth_sighan_2022-03-31_rls.txt"
    eval_file_path = Constants.CSC_RLS_DATA_EVAL_15_DIR

    results = metric_file(pred_path=pred_labels_path, targ_path=true_labels_path, use_decimal=True)

    print("metric_file")
    for k, v in results.items():
        print(f'{k}: {v}')

    # pred_raw_path = f"{file_path}/test.sighan15_predict_result_2022-03-10_rls.txt"
    # pred_raw_path = f"{file_path}/test.sighan15_predict_result_none_post_2022-03-10_rls.txt"
    results = CscMetricDcn.eval_predict(eval_file_path=eval_file_path, predict_result_path=pred_raw_path)
    print(results)

    print("metric_soft_mask_bert")
    for k, v in results["metric_soft_mask_bert"].items():
        print(f'{k}: {v}')

    print("metric_dcn:")
    for k, v in results["metric_dcn"].items():
        print(f"metric_dcn:{k}")
        for key, val in v.items():
            print(f'{key}: {val}')

    results = CscMetricDcn.eval_csc_metric(pred_labels_path=pred_labels_path, true_labels_path=true_labels_path,
                                           pred_raw_path=pred_raw_path, eval_file_path=eval_file_path, show_info=False)
    print(results)

    file_name = f"{Constants.DATA_DIR_JSON_ANALYSIS}/eval_metric_{TimeUtils.get_time()}.json"
    FileUtils.dump_json(file_name, results)


if __name__ == '__main__':
    pass
    demo_metric()
    # eval_file_path = "/home/mqq/shenglei/data/nlp/csc/ReaLiSe/data/trainall.times2.pkl"
    # eval_file_path = "/home/mqq/shenglei/data/nlp/csc/ReaLiSe/data/train_sighan.pkl"
    eval_file_path = "/home/mqq/shenglei/data/nlp/csc/ReaLiSe/data/train_wiki_27.pkl"
    # CscMetricDcn.generate_sighan_format_label_file(eval_file_path=eval_file_path)
