#!/bin/bash

set -v
set -e


begin_time=$(date "+%Y_%m_%d_%H_%M_%S")
echo "train_cls begin time = ${begin_time}"

##########################################################################
## 激活环境
##
##########################################################################

/home/sl/anaconda3/bin/conda init bash
#conda activate pytorch1.9
/home/sl/anaconda3/bin/activate pytorch1.9
#conda list
python --version

##########################################################################
## 变量定义
##
##########################################################################

WORK_DIR=/home/sl/workspace/python/a2022/nlptravel
DATA_DIR="/home/sl/workspace/data/nlp"

TRAIN_FILE="${DATA_DIR}/train.txt"
EVAL_FILE="${DATA_DIR}/dev.txt"
EVAL_LABEL_FILE="${DATA_DIR}/dev_label.txt"
TEST_FILE="${DATA_DIR}/test.txt"


MODEL_NAME="Bert"
#MODEL_NAME_OR_PATH=/home/sl/csc/bert/output/saved_ckpt-83000

NUM_TRAIN_EXAMPLE=281381
BATCH_SIZE=32

SAVE_STEPS=18145
#CALC_STEP=`expr $NUM_TRAIN_EXAMPLE / $BATCH_SIZE  / 10 + 1`
CALC_STEP=1000
EVAL_STEPS=$CALC_STEP
LOG_STEPS=$CALC_STEP
WARMUP_STEPS=10000
SEED=1038
LR=5e-5
SAVE_TOTAL_LIMIT=10
MAX_LENGTH=130
NUM_EPOCHS=10
LOG_DIR=logs

LOG_FILE=${WORK_DIR}"/outputs/logs/run_${MODEL_NAME}_${begin_time}.log"

##########################################################################
## 脚本执行      --add_post_process \
##
##########################################################################
cd ${WORK_DIR}

python $WORK_DIR/finanicial_ner/demo.py

python $WORK_DIR/nlptravel/train_cls.py \
  --model_type ${MODEL_NAME} \
  --task_name csc \
  --model_name ${MODEL_NAME} \
  --output_dir ./${MODEL_NAME} \
  --do_train --do_eval \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --train_data_file $TRAIN_FILE  \
  --eval_data_file $EVAL_FILE \
  --test_data_file $TEST_FILE \
  --eval_label_file $EVAL_LABEL_FILE \
  --num_train_epochs $NUM_EPOCHS \
  --learning_rate $LR \
  --warmup_steps $WARMUP_STEPS \
  --fp16 \
  --fp16_full_eval \
  --evaluation_strategy steps \
  --eval_steps $EVAL_STEPS \
  --save_strategy epoch \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --logging_steps $LOG_STEPS  \
  --metric_for_best_model sent-correct-f1 \
  --show_info > $LOG_FILE  2>&1 &

##########################################################################
echo "end"
end_time=$(date "+%Y_%m_%d_%H_%M_%S")
echo "train_cls end time = ${end_time}"

echo "log file name : ${LOG_FILE}"
echo "tail -f ${LOG_FILE}"

