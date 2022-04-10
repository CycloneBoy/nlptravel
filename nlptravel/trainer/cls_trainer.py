#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel
# @File    ：cls_trainer.py
# @Author  ：sl
# @Date    ：2022/3/9 17:13


import dataclasses
import math
import os
import time
import traceback

import torch.nn.functional as F
import numpy as np
import torch

from torch import nn, optim
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, EvalPrediction, \
    TrainerCallback, is_torch_tpu_available

from transformers.trainer import Trainer
from typing import Optional, List, Union, Callable, Dict, Tuple, Any

from transformers.trainer_utils import speed_metrics, denumpify_detensorize, EvalLoopOutput, PredictionOutput, \
    TrainOutput

from nlptravel.entity.ner_common_entity import ModelArguments, DataTrainingArguments
from nlptravel.loss.rdrop_loss import RDropLoss
from nlptravel.utils.base_utils import ModelNameType

from nlptravel.utils.constant import Constants
from nlptravel.utils.file_utils import FileUtils
from nlptravel.utils.logger_utils import logger
from nlptravel.utils.time_utils import TimeUtils, Timer

"""
CLS trainer

"""


class ClsTrainer(Trainer):
    """
    CLS trainer
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            data_args: DataTrainingArguments = None,
            model_args: ModelArguments = None,
            run_log=None,
    ):
        super().__init__(model=model, args=args, data_collator=data_collator,
                         train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer,
                         model_init=model_init, compute_metrics=compute_metrics, callbacks=callbacks,
                         optimizers=optimizers)

        self.data_args = data_args
        self.model_args = model_args
        self.show_info = self.model_args.show_info
        self.run_log = run_log
        self.output_dir = self.args.output_dir

        ####################################################################
        # rdrop loss
        #
        ####################################################################
        self.custom_rdrop_loss = False
        self.alpha = 4
        self.rdrop_loss = RDropLoss(alpha=self.alpha)

        ####################################################################
        # 自己的训练函数
        #
        ####################################################################
        self.custom_train = False
        if self.model_args.model_name == ModelNameType.CLS_TEXT_CNN.desc:
            self.custom_train = True

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss, outputs = super().compute_loss(model=model, inputs=inputs, return_outputs=True)

        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.custom_rdrop_loss:
            y_pred = outputs[1]
            loss = self.rdrop_loss(y_pred, labels=labels, ce_loss=loss)

        return (loss, outputs) if return_outputs else loss

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            **kwargs,
    ):
        if self.custom_train:
            return self.train_custom()
        else:
            return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial)

    def train_custom(self):

        args = self.args

        self.is_in_train = True

        start_time = time.time()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        total_batch = 0  # 记录进行到多少batch

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Train!
        num_examples = self.num_examples(train_dataloader)
        num_train_epochs = args.num_train_epochs
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        max_steps = math.ceil(args.num_train_epochs * 1)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        for epoch in range(int(args.num_train_epochs)):
            print('Epoch [{}/{}]'.format(epoch + 1, args.num_train_epochs))
            # scheduler.step() # 学习率衰减

            for step, batch in tqdm(enumerate(train_dataloader)):
                self.model.train()

                for k, v in batch.items():
                    batch[k] = v.to(self.args.device)

                outputs = self.model(**batch)
                self.model.zero_grad()

                loss = outputs[0]

                loss.backward()
                optimizer.step()

                if total_batch % args.eval_steps == 0:
                    eval_output = self.evaluate()
                    logger.info(f"评估指标： {eval_output}")

                    self.model.train()
                total_batch += 1

            logger.info(f"epoch: {epoch} / {int(args.num_train_epochs)} finish")
            eval_output = self.evaluate()
            logger.info(f"评估指标： {eval_output}")

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_metrics = None
        eval_begin_time = TimeUtils.now_str()

        if self.custom_train:
            eval_metrics = self.evaluate_custom(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                                metric_key_prefix=metric_key_prefix)
        else:
            # 默认的评估过程
            eval_metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                            metric_key_prefix=metric_key_prefix)

            self.save_eval_predict_to_dest(eval_metrics)

        # add save the eval metric to db
        eval_end_time = TimeUtils.now_str()

        ######################################################################
        # 指标汇总

        return eval_metrics

    def save_eval_predict_to_dest(self, eval_metrics):
        """
        保存推理过程的预测结果

        :param eval_metrics:
        :return:
        """
        current_step = self.state.global_step
        #  拷贝文件到
        output_dir = os.path.join(self.output_dir, "eval_metric")
        epoch = self.state.epoch if self.state.epoch is not None else 0.0

        eval_result_path = f"{output_dir}/eval_{epoch:.2f}_{current_step}_metric.json"
        eval_metrics["step"] = current_step
        FileUtils.dump_json(eval_result_path, eval_metrics)

        file_dir = f"{Constants.DATA_DIR}/test"
        cls_predict_result_path = f"{file_dir}/cls_predict.txt"
        cls_char_predict_result_path = f"{file_dir}/cls_char_predict.txt"
        predict_result_path = f"{file_dir}/cls_detect_predict.txt"
        label_result_path = f"{file_dir}/cls_detect_label.txt"
        eval_metric_file = f"{file_dir}/cls_sighan15_metrics.json"

        eval_file_output_dir = f"{self.output_dir}/eval_predict/eval_{int(epoch)}"
        save_time = TimeUtils.now_str_short()

        self.copy_file_to_eval(file_name=cls_predict_result_path, eval_file_output_dir=eval_file_output_dir,
                               save_time=save_time)
        self.copy_file_to_eval(file_name=cls_char_predict_result_path, eval_file_output_dir=eval_file_output_dir,
                               save_time=save_time)
        self.copy_file_to_eval(file_name=predict_result_path, eval_file_output_dir=eval_file_output_dir,
                               save_time=save_time)
        self.copy_file_to_eval(file_name=label_result_path, eval_file_output_dir=eval_file_output_dir,
                               save_time=save_time)
        self.copy_file_to_eval(file_name=eval_metric_file, eval_file_output_dir=eval_file_output_dir,
                               save_time=save_time, endwith=".json")

    def evaluate_custom(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        # memory metrics - must set up as early as possible
        # self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        eval_begin_time = TimeUtils.now_str()

        eval_output_dir = self.args.output_dir
        if not os.path.exists(eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # Eval!
        num_samples = len(self.eval_dataset)
        logger.info("***** Running evaluation %s *****", metric_key_prefix)
        logger.info("  Num examples = %d", num_samples)
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        run_data_total = 0
        pbar = tqdm(total=len(eval_dataloader), desc='Evaluating')
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module

        all_labels = []
        all_predicts = []
        for step, batch in enumerate(eval_dataloader):
            self.model.eval()
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.args.device)
                # for k, v in batch.items():
                #     batch[k] = v.to(self.args.device)

                # outputs = self.model(**batch)
                try:
                    outputs = self.model(input_ids=input_ids)
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"batch: {batch}")
                    logger.error(f"outputs: {outputs}")

                # tmp_eval_loss = outputs[0]
                logits = outputs[0]

                labels = batch["labels"].detach().cpu().numpy().tolist()
                logits_max = torch.max(logits, 1)
                predicts = torch.max(logits, 1)[1].cpu().numpy().tolist()

                eval_loss += 0
                run_data_total += len(batch["input_ids"])

                all_labels.extend(labels)
                all_predicts.extend(predicts)

            # 显示 进度
            global_avg_loss = eval_loss / step if step > 0 else 0
            pbar.set_postfix({'loss': global_avg_loss})
            pbar.update()
        pbar.close()
        logger.info("\n")

        # 计算指标
        eval_loss = eval_loss / run_data_total

        all_predicts = np.ndarray(all_predicts)
        all_labels = np.ndarray(all_labels)

        results = self.compute_metrics(EvalPrediction(predictions=all_predicts, label_ids=all_labels))

        logger.info(f"整体评估指标: {results}")

        # 保存评估指标
        all_metrics = results
        results["loss"] = eval_loss

        eval_end_time = TimeUtils.now_str()

        # 按照 trainer 格式 处理后续结果
        metrics = denumpify_detensorize(results)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # return results
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )

        self.log(metrics)
        ######################################################################
        # 指标汇总：
        run_result_metric_file = f"{Constants.DATA_DIR_JSON_RUN_METRIC}/{TimeUtils.get_time()}/eval_metric_{TimeUtils.now_str_short()}.json"

        all_metric = {
            "model_name": self.model_args.model_name,
            "task_name": self.model_args.task_name,
            "model_name_or_path": self.model_args.model_name_or_path,
            "eval_file_path": [self.data_args.eval_data_file],
            "eval_begin_time": eval_begin_time,
            "eval_end_time": eval_end_time,
            "use_time": TimeUtils.calc_diff_time(eval_begin_time, eval_end_time),
            "run_metric_validate": {
                "eval_metric": metrics,
                "metric": {
                    "loss": eval_loss,
                    "all_metrics": all_metrics
                },
                "eval_info": {
                    "total_length": num_samples,
                    "eval_batch_size": self.args.eval_batch_size,
                    "type_weight": self.eval_dataset.type_weight,
                }
            },
            "metric_file": run_result_metric_file,
        }

        FileUtils.dump_json(run_result_metric_file, all_metric)
        logger.info(f"all_metric: {all_metric}")

        # 保存评估指标到数据库
        # self.save_metric_to_db(all_metric=all_metric, need_save_file=True, run_log=["命名实体识别"])
        ######################################################################

        # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        # self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def copy_file_to_eval(self, file_name, eval_file_output_dir, save_time, endwith=".txt"):
        """
        拷贝文件到指定目录

        :param file_name:
        :param eval_file_output_dir:
        :param save_time:
        :param endwith:
        :return:
        """
        epoch = self.state.epoch if self.state.epoch is not None else 0.0
        current_step = self.state.global_step

        save_file_name = f"{eval_file_output_dir}/eval_{FileUtils.get_file_name(file_name)}_{epoch:.2f}_{current_step}_{save_time}{endwith}"

        FileUtils.copy_file_rename(file_name, save_file_name)