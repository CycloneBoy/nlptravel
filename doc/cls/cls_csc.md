# cls_csc

## checkpoint
/home/sl/workspace/data/nlp/hfl/chinese-electra-small-discriminator
"hfl/chinese-electra-180g-small-ex-discriminator"


## 

```shell
--model_type
bert
--task_name
cls_slot
--dataset_name
csc_rls
--model_name
AutoModelForTokenClassification
--output_dir
./JointBERT
--do_eval
--per_device_train_batch_size
64
--per_device_eval_batch_size
64
--model_name_or_path
"hfl/chinese-electra-180g-small-discriminator"
--num_train_epochs
10
--learning_rate
5e-5
--warmup_steps
10000
--fp16
--fp16_full_eval
--evaluation_strategy
steps
--eval_steps
1000
--save_strategy
epoch
--save_total_limit
10
--logging_steps
1000
--metric_for_best_model
f1
--show_info
--loss_weight
0
```