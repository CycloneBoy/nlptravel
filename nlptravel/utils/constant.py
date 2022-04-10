#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel
# @File    ：constant.py
# @Author  ：sl
# @Date    ：2021/11/8 14:33
import re


class Constants(object):
    """
    常量工具类

    """
    # WORK_DIR = "./"

    WORK_DIR = "/home/sl/workspace/python/a2022/nlptravel"

    # 数据保存目录
    DATA_DIR = f"{WORK_DIR}/data"
    LOG_DIR = f"{WORK_DIR}/logs"

    # 日志相关
    LOG_FILE = f"{LOG_DIR}/run.log"
    LOG_LEVEL = "debug"

    NLP_DATA_DIR = f"/home/sl/workspace/data/nlp"
    NLP_CSC_DATA_DIR = f"{NLP_DATA_DIR}/csc"

    BASE_DIR = WORK_DIR
    SRC_DIR = f"{WORK_DIR}/nlptravel"
    DIR_SRC_DATA = f"{SRC_DIR}/data"
    # 配置文件路径
    CONFIG_DIR = f"{DIR_SRC_DATA}/conf"
    # 常用文件路径
    DIR_DATA_COMMON = f"{DIR_SRC_DATA}/common"
    # feature文件路径
    DIR_DATA_FEATURE = f"{DIR_SRC_DATA}/feature"

    # text
    DIR_DATA_TEXT = f"{DIR_SRC_DATA}/text"

    DIR_DATA_JSON_ANALYSIS = f"{DATA_DIR}/json/analysis"

    # BERT_PATH
    NLP_PRETRAIN_DIR = NLP_DATA_DIR

    # pd 4列名称
    COLUMN_NAME_2 = ['text', 'label']
    COLUMN_NAME_4 = ['text', 'label', 'mask', 'pinyin']

    # 分隔符
    DELIMITER_TAB = "\t"

    ####################################################################################
    # Plm 模型
    #
    ####################################################################################

    BERT_VOCAB_FILE = f"{NLP_DATA_DIR}/chinese_roberta_wwm_ext_pytorch/vocab.txt"
    ELECTRA_SAMLL_DISCRIMINATOR = f"{NLP_DATA_DIR}/hfl/chinese-electra-small-discriminator"

    ####################################################################################
    # CSC ReadLiSee
    #
    ####################################################################################
    # csc RLS
    DATASET_NAME_CLS_RLS = "csc_rls"

    CSC_RLS_RAW_CHECKPOINT_DIR = f"{NLP_CSC_DATA_DIR}/ReaLiSe"

    CSC_RLS_DATA_DIR = f"{CSC_RLS_RAW_CHECKPOINT_DIR}/data"
    CSC_RLS_DATA_TRAIN_DIR = f"{CSC_RLS_DATA_DIR}/trainall.times2.pkl"
    CSC_RLS_DATA_TRAIN_DIR_CSV = f"{CSC_RLS_DATA_DIR}/trainall.times2.csv"
    CSC_RLS_DATA_TRAIN_DIR_GEN = f"{CSC_RLS_DATA_DIR}/trainall.times2_gen.pkl"
    CSC_RLS_DATA_TRAIN_SIGHAN_DIR = f"{CSC_RLS_DATA_DIR}/train_sighan.pkl"
    CSC_RLS_DATA_TRAIN_SIGHAN_DIR_GEN = f"{CSC_RLS_DATA_DIR}/train_sighan_gen.pkl"
    CSC_RLS_DATA_TRAIN_SIGHAN_DIR_CSV = f"{CSC_RLS_DATA_DIR}/train_sighan.csv"
    CSC_RLS_DATA_TRAIN_WIKI_27_DIR = f"{CSC_RLS_DATA_DIR}/train_wiki_27.pkl"

    CSC_RLS_DATA_EVAL_15_DIR = f"{CSC_RLS_DATA_DIR}/test.sighan15.pkl"
    CSC_RLS_DATA_EVAL_LABEL_15_DIR = f"{CSC_RLS_DATA_DIR}/test.sighan15.lbl.tsv"
    CSC_RLS_DATA_EVAL_15_DIR_CSV = f"{CSC_RLS_DATA_DIR}/test_sighan15.csv"
    CSC_RLS_DATA_EVAL_15_CLS_LABEL_DIR_CSV = f"{CSC_RLS_DATA_DIR}/test_sighan15_cls_label.csv"

    CSC_RLS_DATA_EVAL_14_DIR = f"{CSC_RLS_DATA_DIR}/test.sighan14.pkl"
    CSC_RLS_DATA_EVAL_LABEL_14_DIR = f"{CSC_RLS_DATA_DIR}/test.sighan14.lbl.tsv"

    CSC_RLS_DATA_EVAL_13_DIR = f"{CSC_RLS_DATA_DIR}/test.sighan13.pkl"
    CSC_RLS_DATA_EVAL_LABEL_13_DIR = f"{CSC_RLS_DATA_DIR}/test.sighan13.lbl.tsv"

    ####################################################################################
    # CLS 分类模型
    #
    ####################################################################################
    CLS_RLS_DATA_TRAIN_DIR_CSV = f"{NLP_DATA_DIR}/cls_train.csv"

    CLS_RLS_DATA_EVAL_15_CSV = f"{NLP_DATA_DIR}/cls_test_sighan15.csv"

    CSC_DATA_WIKI_GEN = f"{NLP_DATA_DIR}/csc/gen_data/wiki_train_gen_dcn.csv"
    CLS_DATA_WIKI_GEN = f"{NLP_DATA_DIR}/cls/data/wiki_train_gen_dcn.csv"

    CLS_DATA_TRAIN_ALL = f"{NLP_DATA_DIR}/cls/data/train_all.csv"
    CLS_DATA_TRAIN_DEV = f"{NLP_DATA_DIR}/cls/data/train_dev.csv"

    # tnews_public
    DATASET_NAME_CLS_TNEWS = "tnews_public"
    DATASET_NAME_CLS_TNEWS_LABELS = ["100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112",
                                     "113", "114", "115", "116"]
    CLS_DATA_TNEWS_TRAIN = f"{NLP_DATA_DIR}/tnews_public/train.json"
    CLS_DATA_TNEWS_DEV = f"{NLP_DATA_DIR}/tnews_public/dev.json"
    CLS_DATA_TNEWS_TEST = f"{NLP_DATA_DIR}/tnews_public/test.json"

    # RLS 中间输出文件路径
    CSC_RLS_DATA_OUTPUT_DIR = f"{CSC_RLS_RAW_CHECKPOINT_DIR}/output"

    # cls_slot
    DATASET_NAME_CLS_SLOT = "cls_slot"

    CSC_DATA_TRAIN_28_WIKI_DIR_CSV = f"{NLP_CSC_DATA_DIR}/train.txt"
    CSC_DATA_TRAIN_28_WIKI_CLS_LABEL_DIR_CSV = f"{NLP_CSC_DATA_DIR}/train_cls_label.csv"

    CSC_DATA_EVAL_15_DIR_CSV = f"{NLP_CSC_DATA_DIR}/sighan15_test.txt"
    CSC_DATA_EVAL_15_CLS_LABEL_DIR_CSV = f"{NLP_CSC_DATA_DIR}/sighan15_test_cls_label.csv"

    CSC_RLS_DATA_EVAL_15_CLS_LABEL_DIR_CSV = CSC_DATA_EVAL_15_CLS_LABEL_DIR_CSV

    ####################################################################################
    # 命名实体识别 NER
    #
    ####################################################################################

    NER_CLUENER_DATASET_DIR = f"{NLP_DATA_DIR}/ner/public/cluener"

    ####################################################################################
    # EDA
    #
    ####################################################################################

    STOPWORDS_PATH = f"{DIR_DATA_TEXT}/stopwords/hit_stopwords.txt"
    CONFUSION_SET_PATH = f"{DIR_DATA_COMMON}/confusion.txt"
    # 同义词
    SAME_MEAN_WORD_PATH = f"{DIR_DATA_TEXT}/eda/同义词.txt"
