#!/usr/bin/env bash
TASK_NAME="cmnli"
# MODEL_NAME="./cmnli_output/bert/"
# MODEL_NAME="../pretrained_lm/chinese_roberta_wwm_ext_pytorch/"
MODEL_NAME="../pretrained_lm/macbert_large/"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"

export FewCLUE_DATA_DIR=../../../datasets/

# make output dir
# if [ ! -d $CURRENT_DIR/${TASK_NAME}_output ]; then
#   mkdir -p $CURRENT_DIR/${TASK_NAME}_output
#   echo "makedir $CURRENT_DIR/${TASK_NAME}_output"
# fi

# run task
cd $CURRENT_DIR
echo "Start running..."
if [ $# == 0 ]; then
   nohup python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_intermediate_train \
      --do_lower_case \
      --data_dir=$FewCLUE_DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=128 \
      --per_gpu_train_batch_size=32 \
      --per_gpu_eval_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=2 \
      --logging_steps=3335 \
      --save_steps=1000 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output_bert/ \
      --overwrite_output_dir \
      --seed=42 \
      --bert_multi &
fi
