#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：financial_ner 
# @File    ：ner_common_entity.py
# @Author  ：sl
# @Date    ：2022/1/18 16:35
import copy
import json
from dataclasses import field, dataclass
from typing import Optional, Dict, Any, List

from nlptravel.utils.base_utils import NlpPretrain, NlpTaskType, DataFileType, ModelNameType
from nlptravel.utils.constant import Constants

"""
NER common entity
"""


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    show_info: bool = field(default=False, metadata={"help": "show_info"})
    use_lstm: bool = field(default=False, metadata={"help": "是否使用LSTM"})
    soft_label: bool = field(default=False, metadata={"help": "是否使用span"})
    embedding_vocab_size: int = field(default=10000, metadata={"help": "embedding_vocab_size"})
    embedding_size: int = field(default=300, metadata={"help": "embedding_size"})
    embedding_pretrained: str = field(default=None, metadata={"help": "embedding_pretrained"})
    lstm_hidden_size: int = field(default=500, metadata={"help": "LSTM隐藏层输出的维度"})
    lstm_layers: int = field(default=1, metadata={"help": "堆叠LSTM的层数"})
    lstm_dropout: float = field(default=0.5, metadata={"help": "LSTM的dropout"})
    hidden_dropout: float = field(default=0.5, metadata={"help": "预训练模型输出向量表示的dropout"})
    ner_num_labels: int = field(default=34, metadata={"help": "需要预测的标签数量"})
    loss_type: str = field(default="ce", metadata={"help": "损失函数类型，['ce', 'bce', 'dice', 'focal', 'adaptive_dice']"})
    loss_ignore_index: int = field(default=-11, metadata={"help": "loss_ignore_index"})
    loss_reduction: str = field(default="default", metadata={"help": "loss_reduction:mean"})

    activate_func: str = field(default="gelu", metadata={"help": "激活函数类型"})
    mrc_span_loss_candidates: str = field(default="gold_pred_random",
                                          metadata={"help": "Candidates used to compute span loss"})
    mrc_pred_answerable: bool = field(default=True, metadata={"help": "bert mrc 返回答案"})
    mrc_construct_entity_span: str = field(default="start_end_match",
                                           metadata={
                                               "help": "bert mrc模型实体匹配方法：start_end_match,match,start_and_end,start_end"})

    ## focal loss
    focal_gamma: float = field(default=2, metadata={"help": "focal_gamma"})

    ## dice loss
    dice_smooth: float = field(default=1, metadata={"help": "smooth value of dice loss"})
    dice_ohem: float = field(default=0.3, metadata={"help": "ohem ratio of dice loss"})
    dice_alpha: float = field(default=0.01, metadata={"help": "alpha value of adaptive dice loss"})
    dice_square: bool = field(default=True, metadata={"help": "use square for dice loss"})

    ##  loss
    weight_start: float = field(default=1.0, metadata={"help": "weight_start"})
    weight_end: float = field(default=1.0, metadata={"help": "weight_end"})
    weight_span: float = field(default=0.2, metadata={"help": "weight_span"})
    answerable_task_ratio: float = field(default=0.2, metadata={"help": "answerable_task_ratio"})
    loss_dynamic: bool = field(default=True, metadata={"help": "动态计算loss "})

    use_return_dict: bool = field(default=False, metadata={"help": "是否返回dict "})
    add_post_process: bool = field(default=False, metadata={"help": "是否添加后处理 "})
    post_process_type: str = field(default="", metadata={"help": "后处理类型"})

    eval_ture_input: bool = field(default=False, metadata={"help": "是否采用完全正确的句子输入"})
    extract_svm_feature: bool = field(default=False, metadata={"help": "提取SVM特征"})

    use_rdrop_loss: bool = field(default=False, metadata={"help": "是否采用RDropLoss"})

    ## 分类
    num_labels: int = field(default=2, metadata={"help": "需要预测的标签数量"})
    num_filters: int = field(default=256, metadata={"help": "num_filters"})
    filter_sizes: str = field(default="2,3,4", metadata={"help": "filter_sizes"})
    loss_weight: float = field(default=1.0, metadata={"help": "loss_weight"})
    early_stopping_patience: int = field(default=-1, metadata={"help": "early_stopping_patience"})
    early_stopping_threshold: float = field(default=0.005, metadata={"help": "early_stopping_threshold"})

    model_name_or_path: Optional[str] = field(
        default=NlpPretrain.ROBERTA_CHINESE_WWM_EXT_PYTORCH.path,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    model_type: Optional[str] = field(
        default="bert",
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(["bert"])},
    )

    model_name: str = field(default="bert_softmax", metadata={"help": "模型名称"})
    task_name: str = field(default="cluener", metadata={"help": "任务类型"})

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default="../data/pretrained_models",
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)

        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(default="cls", metadata={"help": "数据集类型"})

    train_data_file: Optional[str] = field(
        default=Constants.NER_CLUENER_DATASET_DIR + "/train.json",
        metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=Constants.NER_CLUENER_DATASET_DIR + "/dev.json",
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    test_data_file: Optional[str] = field(
        default=Constants.NER_CLUENER_DATASET_DIR + "/test.json",
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    eval_label_file: Optional[str] = field(
        default=Constants.NLP_CSC_DATA_DIR + "/test.sighan15.lbl.tsv",
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    vocab_file: Optional[str] = field(
        default=Constants.NER_CLUENER_DATASET_DIR + "/vocab.pkl",
        metadata={
            "help": "vocab_file ."},
    )

    block_size: int = field(
        default=128,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    output_file: Optional[str] = field(
        default=Constants.WORK_DIR + "/outputs",
        metadata={"help": "The output data file."},
    )

    batch_size: int = field(
        default=24,
        metadata={"help": "batch size for train"},
    )

    testset_year: int = field(
        default=15,
        metadata={"help": "sighan eval year: 13,14,15"},
    )


@dataclass
class ExtractNerArguments:
    """
    提取 ner 的参数

    """

    file_name: str = field(default="./test.txt", metadata={"help": "file_name"})
    column_name: str = field(default="text,label,mask,pinyin", metadata={"help": "column_name"})
    task_type: NlpTaskType = field(default=NlpTaskType.NER, metadata={"help": "task_type"})
    use_cuda: bool = field(default=True, metadata={"help": "use_cuda"})
    show_info: bool = field(default=True, metadata={"help": "show_info"})
    use_large_model: bool = field(default=True, metadata={"help": "use_large_model"})
    need_cache: bool = field(default=True, metadata={"help": "need_cache"})
    data_type: DataFileType = field(default=DataFileType.CSV, metadata={"help": "use_cuda"})

    show_number: int = field(default=10, metadata={"help": "show_number"})
    model_name_type: ModelNameType = field(default=ModelNameType.NER_FASTHAN, metadata={"help": "model_name_type"})
