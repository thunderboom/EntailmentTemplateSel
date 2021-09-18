# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-12-30 19:26:53
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-31 19:49:36
""" Finetuning the library models for sequence classification on CLUE (Bert, ERNIE, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import json
import numpy as np
import pdb
import torch
import copy
from scipy.special import softmax
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from model import BertMulti
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          AlbertForSequenceClassification,
                          BertModel, BertForMaskedLM)

from transformers import AdamW, WarmupLinearSchedule
from metrics.clue_compute_metrics import compute_metrics, simple_accuracy
from processors import clue_output_modes as output_modes
from processors.efl_fewclue import clue_processors as processors
from processors import clue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn, xlnet_collate_fn
from tools.common import seed_everything, save_numpy
from tools.common import init_logger, logger
from tools.progressbar import ProgressBar
from task_label_description import (
        tnews_label_descriptions,
        eprstmt_label_descriptions,
        csldcp_label_description,
        iflytek_label_description,
        bustm_label_description,
        ocnli_label_description,
        chid_label_description,
        csl_label_description,
        cluewsc_label_description,
        )

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig,
                                                                                RobertaConfig)), ())
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'albert': (BertConfig, AlbertForSequenceClassification, BertTokenizer)
}

TASK_LABELS_DESC={
        "tnews":tnews_label_descriptions,
        "eprstmt":eprstmt_label_descriptions,
        "csldcp":csldcp_label_description,
        "iflytek":iflytek_label_description,
        "bustm":bustm_label_description,
        "ocnli":ocnli_label_description,
        "chid":chid_label_description,
        "csl":csl_label_description,
        "cluewsc":cluewsc_label_description,
        "cmnli":ocnli_label_description,
        }

def get_score(language_model, inputs_ids, token_type_ids):
    predictions = language_model(inputs_ids)  # model(masked_ids)
    B, L, D = predictions[0].size()[:3]
    loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
    predictions = predictions[0].view(B*L, D)
    token_type_ids = token_type_ids.view(B*L)
    label = inputs_ids.view(B*L)
    loss = loss_fct(predictions, label).data#已经取平均值后的loss，作为句子的ppl分数返回
    loss = loss * token_type_ids
    loss = loss.sum() / token_type_ids.sum()
    loss = loss.cpu().item()
    return loss
    
def get_template_score(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
    score_list = []
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]}
            b_score = get_score(model, inputs['input_ids'], inputs['token_type_ids'])
            score_list.append(b_score)
            pbar(step)
    logger.info("template score:{}".format(np.mean(score_list)))



def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if task in ['mnli', 'mnli-mm'] and 'roberta' in args.model_type:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]

    task_label_description=TASK_LABELS_DESC[args.task_name]

    if data_type == 'train':
        examples,test_sentences_labels = processor.get_train_examples(args.data_dir,task_label_description, args.task_idx)
    elif data_type == 'dev':
        examples,test_sentences_labels = processor.get_dev_examples(args.data_dir,task_label_description, args.task_idx)
    else:
        examples,test_sentences_labels = processor.get_test_examples(args.data_dir,task_label_description)

    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_token_ids = torch.tensor([f.token_idx for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels, all_token_ids)
    return dataset,test_sentences_labels


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # add
    parser.add_argument("--task_idx", type=str, default="0",
                        help="Whether to run training.")
    parser.add_argument("--bert_multi", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_intermediate_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--load_cmnli_model", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--cmnli_path", default='./cmnli_output/bert/', type=str,
                        help="cmnli path")
    parser.add_argument("--bootStrap", action='store_true',
                        help='bootStrapping tag for training')
    parser.add_argument("--num_boot_epochs", default=5, type=int,
                        help="boot training epochs.")
    
    ## Other parameterss
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    seed_everything(args.seed)
    # Prepare CLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    logger.info("evaluation parameters %s", args)
    # Training
    train_dataset,_ = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
    get_template_score(args, train_dataset, model, tokenizer)


if __name__ == "__main__":
    main()
