#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel
# @File    ：base_utils.py
# @Author  ：sl
# @Date    ：2021/12/3 10:55

"""
工具类基类

"""
import os
from abc import abstractmethod, ABC
from enum import Enum, unique

from nlptravel.utils.constant import Constants


class BaseUtil(ABC):
    """
    抽取数据基类
    """

    @abstractmethod
    def init(self):
        """
        工具类初始化
        :return:
        """
        pass


@unique
class CscErrorType(Enum):
    NONE = "未知"
    SUCCESS = "正确"
    OUT_OF_ORDER = "语序"
    NUMBER = "数字"
    MISS = "漏纠"
    MISS_WORD = "漏纠-词组"
    ERROR = "误纠"
    FAULT = "过纠"
    FAULT_SINGLE = "过纠-单字"
    FAULT_MULTI_CHAR = "过纠-多字"
    FAULT_MULTI_WORD = "过纠-连词"
    FAULT_MULTI_WORD_NER = "过纠-专有名词"
    FAULT_SIMILAR_STROKE = "过纠-近字形"
    FAULT_SIMILAR_PINYIN = "过纠-近拼音"
    FAULT_NUMBER = "过纠-数字"
    NER_NAME = "专有名词-名称"
    SIMILAR_PINYIN = "近拼音错误"
    SIMILAR_STROKE = "近字形错误"
    SIMILAR_PINYIN_STROKE = "近拼音字形错误"
    RANDOM = "随机错误"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc


@unique
class NlpTaskType(Enum):
    """
    nlp 任务类型
    """
    PARSING = "依存分析"
    CWS = "分词"
    POS = "词性标注"
    NER = "ner"
    CSC = "纠错"
    WORD_FREQUENCY = "词频"

    NONE = "未知"
    DEFAULT = "默认"
    PADDLE_NLP = "PaddleNlp"

    # LTP
    LTP_SENT_SPLIT = "分句"
    LTP_SRL = "语义角色标注"
    # LTP_DEP = "依存句法分析"
    LTP_SDP_TREE = "语义依存分析(树)"
    LTP_SDP_GRAPH = "语义依存分析(图)"

    # PaddleNlp
    PADDLENLP_WORD_SEGMENTATION = "word_segmentation"
    PADDLENLP_POS_TAGGING = "pos_tagging"
    PADDLENLP_CSC = "text_correction"
    # PADDLENLP_NER = "ner"
    PADDLENLP_DEPENDENCY_PARSING = "dependency_parsing"
    PADDLENLP_SENTIMENT_ANALYSIS = "sentiment_analysis"
    PADDLENLP_TEXT_SIMILARITY = "text_similarity"
    PADDLENLP_KNOWLEDGE_MINING = "knowledge_mining"
    PADDLENLP_QUESTION_ANSWERING = "question_answering"
    PADDLENLP_POETRY_GENERATION = "poetry_generation"
    PADDLENLP_DIALOGUE = "dialogue"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc

    @staticmethod
    def parse(task_type):
        for name, member in NlpTaskType.__members__.items():
            if str(task_type).lower() == str(name).lower():
                return member
        return NlpTaskType.NONE


@unique
class PaddleNlpTaskType(Enum):
    """
    PaddleNlp 任务类型
        - https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/taskflow.md#%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB
    """
    WORD_SEGMENTATION = "word_segmentation"
    POS_TAGGING = "pos_tagging"
    NER = "ner"
    CSC = "text_correction"
    DEPENDENCY_PARSING = "dependency_parsing"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_SIMILARITY = "text_similarity"
    KNOWLEDGE_MINING = "knowledge_mining"
    QUESTION_ANSWERING = "question_answering"
    POETRY_GENERATION = "poetry_generation"
    DIALOGUE = "dialogue"

    DEFAULT = "默认"
    PADDLE_NLP = "PaddleNlp"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc


class CscModel(ABC):
    """
    CSC 模型基类
    """

    @abstractmethod
    def init(self):
        """
        工具类初始化
        :return:
        """
        pass


def build_path(name):
    return os.path.join(Constants.NLP_PRETRAIN_DIR, name)


@unique
class NlpPretrain(Enum):
    """
    NLP 预训练模型
    """

    def __init__(self, path, description):
        self.path = path
        self.description = description

    BERT_BASE_UNCASED = (build_path('bert-base-uncased'), 'bert')
    BERT_BASE_CHINESE = (build_path('bert-base-chinese'), 'bert')
    BERT_CHINESE_WWM = (build_path('hfl/chinese-bert-wwm'), 'bert-wwm')
    BERT_CHINESE_WWM_EXT = (build_path('hfl/chinese-bert-wwm-ext'), 'bert-wwm')
    ROBERTA_CHINESE_WWM_EXT_PYTORCH = (build_path('chinese_roberta_wwm_ext_pytorch'), 'roberta')
    ROBERTA_CHINESE_WWM_LARGE_EXT_PYTORCH = (build_path('chinese_roberta_wwm_large_ext_pytorch'), 'roberta')

    ELECTRA_CHINESE_SMALL_GENERATOR = (build_path('hfl/chinese-electra-small-generator'), 'electra')
    ELECTRA_CHINESE_SMALL_DISCRIMINATOR = (build_path('hfl/chinese-electra-small-discriminator'), 'electra')
    ELECTRA_CHINESE_180G_SMALL_DISCRIMINATOR = (build_path('hfl/chinese-electra-180g-small-discriminator'), 'electra')
    ELECTRA_CHINESE_180G_BASE_DISCRIMINATOR = (build_path('hfl/chinese-electra-180g-base-discriminator'), 'electra')
    ALBERT_CHINESE_TINY = (build_path('voidful/albert_chinese_tiny'), 'albert')
    CHINESE_BERT_SHANNON_AI = (build_path('chinese-bert'), 'bert')

    ERNIE = (build_path('ernie'), 'ernie')

    def __str__(self):
        return "{}:{}:{}".format(self.name, self.path, self.description)


@unique
class DataFileType(Enum):
    """
    数据类型
    """
    NER_JSON_LINE = "ner json line "
    NER_JSON_LINE_WITHOUT_LABEL = "ner json line without label"
    NER_JSON = "ner json "
    JSON = "json"
    JSON_LINE = "json line"
    NER_TEXT = "NER 常见标注 "
    NER_VERTICAL_BIO = "NER垂直句子BIO标注"
    TEXT = "文本"
    DB_CSV = "数据库csv"
    CSV = "csv"
    CSC_DCN_DB_CSV = "CSC DCN 数据库csv"
    CSC_RELISE_PICKLE_FILE = "CSC RELISE pickle file "
    MODEL_CHECKPOINT_DICT = "model checkpoint bin"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc


@unique
class ModelNameType(Enum):
    """
    模型名称类型
    """
    NER_BERT_SOFTMAX = "BertSoftmaxForNer"
    NER_BERT_CRF = "BertCrfForNER"
    NER_BERT_SPAN = "BertSpanForNer"
    NER_BERT_MRC = "BertMrcForNer"
    NER_BILSTM_CRF = "BiLstmCrfForNer"
    NER_FASTHAN = "fasthan"
    NER_LTP = "ltp"
    NER_PADDLE_NLP = "paddle_nlp"

    CSC_RLS = "SpellBertPho2ResArch3"

    CSC_RLS_ELECTRA = "SpellBertPho2ResArch3ELECTRA"
    CSC_RLS_ELECTRA_DCN = "SpellBertPho2ResArch3ElectraDcn"
    CSC_PLOME = "PlomeModel"
    CSC_PLOME_V2 = "BertPlome"
    CSC_BERT_ECOPO = "SpellBertEcopo"
    CSC_RLS_PLOME = "RlsPlome"
    CSC_MAC_BERT_ECOPO = "SpellMacBertEcopo"
    CSC_MAC_BERT_AD_ECOPO = "SpellBertAdEcopo"
    CSC_DCN = "DCNForMaskedLM"
    CSC_RLS_ECOPO = "RlsEcopo"

    CLS_TEXT_CNN = "TextCnn"
    CLS_JOINT_BERT = "JointBERT"
    CLS_AUTOMODEL_FOR_TOKEN_CLASSIFICATION = "AutoModelForTokenClassification"

    NONE = "none"

    def __init__(self, desc):
        self._desc = desc

    @property
    def desc(self):
        return self._desc

    @staticmethod
    def parse(model_name_type):
        for name in ModelNameType:
            # for name, member in ModelNameType.__members__.items():
            if str(model_name_type).lower() == name.desc.lower():
                return name
        return ModelNameType.NONE

    @staticmethod
    def is_bert_cluener_model(model_type):
        """
        判断是否是cluener 模型

        :param model_type:
        :return:
        """
        flag = False
        if model_type == ModelNameType.NER_BERT_SOFTMAX \
                or model_type == ModelNameType.NER_BERT_CRF \
                or model_type == ModelNameType.NER_BERT_SPAN \
                or model_type == ModelNameType.NER_BILSTM_CRF:
            flag = True

        return flag

    @staticmethod
    def is_plome_model(model_type):
        flag = False
        if model_type == ModelNameType.CSC_PLOME.desc \
                or model_type == ModelNameType.CSC_RLS_PLOME.desc \
                or model_type == ModelNameType.CSC_PLOME_V2.desc:
            flag = True

        return flag
