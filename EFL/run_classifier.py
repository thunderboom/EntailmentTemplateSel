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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from model import BertMulti
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          AlbertForSequenceClassification)

from transformers import AdamW, WarmupLinearSchedule
from metrics.clue_compute_metrics import compute_metrics, simple_accuracy
from processors import clue_output_modes as output_modes
from processors import clue_processors as processors
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

def BootStrappingTraining(args, train_dataset, label_list, model, tokenizer):
    """
    train dataset: few-shot dataset
    test dataset: unlabled dataset
    model: untrained model
    """
    origin_model = copy.deepcopy(model)
    train(args, train_dataset, origin_model, tokenizer)
    for i in range(2):
        sentence_labels, labels_prob = predict(args, origin_model, tokenizer, label_list, prefix="", boot_tag=True)
        # top5
        percent = (i+1) * 5.0
        num_category = int(len(sentence_labels) * percent / 100 / len(label_list))
        categ = list(set(sentence_labels))
        temp_num_dict = {label: num_category for label in categ}   # 类别平衡
        temp_value = [(idx, label, percent) for idx, (label, percent) in enumerate(zip(sentence_labels, labels_prob))]
        temp_value = sorted(temp_value, key=lambda x: x[2],  reverse=True)
        logger.info("*****examples num: {}".format(len(sentence_labels)))
        train_add_idx, train_add_labels = [], []
        for (idx, label, percent) in temp_value:
            if temp_num_dict[label] < 0:
                continue
            temp_num_dict[label] -= 1
            train_add_idx.append(idx)
            train_add_labels.append(label)
        print("added labels:{}".format(' '.join(train_add_labels)))
        all_dataset, test_labels, temp_train_dataset = load_and_cache_examples_boot(args, args.task_name, tokenizer,
                                                                            train_add_idx, train_add_labels, label_list)
        origin_model = copy.deepcopy(model)
        train(args, temp_train_dataset, origin_model, tokenizer, boot_tag=True)   # ublabeled train
        train(args, train_dataset, origin_model, tokenizer)     # normal train
    torch.save(origin_model.state_dict(), os.path.join(args.output_dir, 'pytorch_model_{}.bin'.format(args.task_idx)))


def intermidate_train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Intermidate Train!
    logger.info("***** Running training *****")
    logger.info("  Train Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    best_acc = 0
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(args.num_train_epochs)):
        epoch_loss = 0
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'albert',
                                                                           'roberta'] else None  # XLM, DistilBERT don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:   # gradient accumulate
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            if (step + 1) % args.save_steps == 0:  # evalute model
                acc = evaluate(args, model, tokenizer, prefix='', intermediate_tag=True)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))
                logger.info("optmizer step:{} train loss:{} best acc: {} current acc: {}" \
                            .format(step, epoch_loss, best_acc, acc))
                epoch_loss = 0
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step




def train(args, train_dataset, model, tokenizer, boot_tag=False):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    if not boot_tag:
        train_epochs = args.num_train_epochs
    else:
        train_epochs = args.num_boot_epochs
    for _ in range(int(train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'albert',
                                                                           'roberta'] else None  # XLM, DistilBERT don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(" ")
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.bert_multi:
                        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model_{}.bin'.format(args.task_idx)))
                    else:
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args_{}.bin'.format(args.task_idx)))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(vocab_path=output_dir)
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", intermediate_tag=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset,_ = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn)

        # Eval!
        logger.info("********* Running evaluation {} ********".format(prefix))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'albert',
                                                                               'roberta'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                # if intermediate_tag:   # 第三个结果表示
                #     logits = outputs[2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds_max = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds_max = np.squeeze(preds)
        if intermediate_tag:
            result = simple_accuracy(preds_max, out_label_ids)
            return result
        result = compute_metrics(eval_task, preds_max, out_label_ids)
        results.update(result)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("******** Eval results {} ********".format(prefix))
        for key in sorted(result.keys()):
            logger.info(" dev: %s = %s", key, str(result[key]))
    return results

def predict_entropy_finetune(args, model, tokenizer, label_list, prefix="", boot_tag=False):
    task_label_description=TASK_LABELS_DESC[args.task_name]

    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.output_dir,)
    label_map = {i: label for i, label in enumerate(label_list)}

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset,test_sentences_labels = load_and_cache_examples(args, pred_task, tokenizer, data_type='test')
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset) if args.local_rank == -1 else DistributedSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size,
                                     collate_fn=xlnet_collate_fn
                                     if args.model_type in ['xlnet'] else collate_fn)

        logger.info("******** Running prediction finetune {} ********".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        nb_pred_steps = 0
        preds = None
        pbar = ProgressBar(n_total=len(pred_dataloader), desc="Predicting")
        softmax_function=torch.nn.Softmax(dim=1)
        step=1
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        for _ in range(step):
            for step, batch in enumerate(pred_dataloader):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if (
                            'bert' in args.model_type or 'xlnet' in args.model_type) else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                _, logits = outputs
                loss = entropy_loss(logits)
                loss.backward()
                optimizer.step()
                model.zero_grad()




def entropy_loss(logits):  # entropy loss
    loss = - torch.log(logits) - torch.log(1 - logits)
    loss = loss.mean()
    return loss

def predict(args, model, tokenizer, label_list, prefix="", boot_tag=False):
    task_label_description=TASK_LABELS_DESC[args.task_name]

    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.output_dir,)
    label_map = {i: label for i, label in enumerate(label_list)}

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        if boot_tag:
            pred_dataset, test_sentences_labels = load_and_cache_examples(args, pred_task, tokenizer, data_type='boot')
        else:
            pred_dataset,test_sentences_labels = load_and_cache_examples(args, pred_task, tokenizer, data_type='test')
        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset) if args.local_rank == -1 else DistributedSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size,
                                     collate_fn=xlnet_collate_fn
                                     if args.model_type in ['xlnet'] else collate_fn)

        logger.info("******** Running prediction {} ********".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        nb_pred_steps = 0
        preds = None
        pbar = ProgressBar(n_total=len(pred_dataloader), desc="Predicting")
        softmax_function=torch.nn.Softmax(dim=1)
        for step, batch in enumerate(pred_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if (
                            'bert' in args.model_type or 'xlnet' in args.model_type) else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                # _, logits, _ = outputs[:3]
                _, logits = outputs
            nb_pred_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            pbar(step)
        sentence_labels=[]
        sentence_labels_prob=[]
        if not boot_tag:
            if args.task_name in ["iflytek","csldcp","tnews","eprstmt"]:
                assert len(preds)%len(task_label_description)==0
                for i in range(int(len(preds)/len(task_label_description))):
                    sentence_label=list(task_label_description.keys())[np.argmax(preds[i*len(task_label_description):(i+1)*len(task_label_description),0])]
                    sentence_labels.append(sentence_label)
                assert len(sentence_labels)==int(len(preds)/len(task_label_description))
            elif args.task_name in ["bustm","ocnli","csl","cluewsc"]:
                for i in range(len(preds)):
                    sentence_label=list(task_label_description.keys())[np.argmax(preds[i])]
                    sentence_labels.append(sentence_label)
            elif args.task_name in ["chid"]:
                assert len(preds)%7==0
                for i in range(int(len(preds)/7)):
                    sentence_label=str(np.argmax(preds[i*7:(i+1)*7,0]))
                    sentence_labels.append(sentence_label)
                assert len(sentence_labels)==int(len(preds)/7)

            assert len(sentence_labels)==len(test_sentences_labels)
        else:
            if args.task_name in ["iflytek", "csldcp", "tnews", "eprstmt"]:
                assert len(preds) % len(task_label_description) == 0
                sentence_label = list(task_label_description.keys())
                for i in range(int(len(preds)/len(task_label_description))):
                    temp_prob = softmax(preds[i * len(task_label_description):(i + 1) * len(task_label_description), 0])
                    temp_index = np.argmax(temp_prob)
                    sentence_label_prob = temp_prob[temp_index]
                    sentence_labels.append(sentence_label[temp_index])
                    sentence_labels_prob.append(sentence_label_prob)
            elif args.task_name in ["bustm", "ocnli", "csl", "cluewsc"]:
                sentence_label = list(task_label_description.keys())
                for i in range(len(preds)):
                    temp_prob = softmax(preds[i])
                    temp_index = np.argmax(temp_prob)
                    sentence_label_prob = temp_prob[temp_index]
                    sentence_labels.append(sentence_label[temp_index])
                    sentence_labels_prob.append(sentence_label_prob)
            elif args.task_name in ["chid"]:
                for i in range(int(len(preds) / 7)):
                    temp_prob = softmax(preds[i * 7:(i + 1) * 7, 0])
                    temp_index = np.argmax(temp_prob)
                    sentence_label_prob = temp_prob[temp_index]
                    sentence_labels.append(str(temp_index))
                    sentence_labels_prob.append(sentence_label_prob)
                assert len(sentence_labels) == int(len(preds) / 7)
            logger.info("{} - {}".format(len(sentence_labels), len(test_sentences_labels)))
        if boot_tag:
            return sentence_labels, sentence_labels_prob
        if args.task_name in ['tnews', 'iflytek', 'ocnli', 'chid', 'csl', 'cluewsc']:
            task_name = args.task_name+'f'
        else:
            task_name = args.task_name
        output_submit_file = os.path.join(pred_output_dir, prefix, "{}_predict_{}.json".format(task_name, args.task_idx))
        output_labels_file = os.path.join(pred_output_dir, prefix, "{}_labels_{}".format(task_name, args.task_idx))

        # 保存标签结果
        with open(output_submit_file, "w") as writer:
            for i, pred in enumerate(sentence_labels):
                json_d = {}
                json_d['id'] = i
                if args.task_name=="chid":
                    json_d['answer'] = str(pred)
                else:
                    json_d['label'] = str(pred)
                writer.write(json.dumps(json_d) + '\n')
        # 保存中间预测结果
        with open(output_labels_file,'w') as writer:
            writer.writelines("%s\n" % sentence_label for sentence_label in sentence_labels)

        print("acc is "+str(np.sum(np.array(test_sentences_labels)==np.array(sentence_labels))/len(test_sentences_labels)))

def load_and_cache_examples_boot(args, task, tokenizer, selected_index=None, selected_label=None, labels_list=None):
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
    examples, test_sentences_labels = processor.get_unlabeled_examples(args.data_dir,task_label_description)
    num = len(task_label_description) * 500
    examples, test_sentences_labels = examples[:num], test_sentences_labels[:num]
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
    features_train = []
    for idx_, label in zip(selected_index, selected_label):
        if args.task_name in ["iflytek", "csldcp", "tnews", "eprstmt"]:
            sentence_label = list(task_label_description.keys())
            gd_idx = sentence_label.index(label)
            for j in range(len(sentence_label)):  # entailment
                temp_idx = len(sentence_label) * idx_ + j
                if j == gd_idx:
                    features[temp_idx].label = 0   # entail
                else:
                    features[temp_idx].label = 1   # not_entail
                features_train.append(features[temp_idx])
        elif args.task_name in ["bustm", "ocnli", "csl", "cluewsc"]:
            sentence_label = list(task_label_description.keys())
            gd_idx = labels_list.index(label)
            features[idx_].label = gd_idx  # not_entail
            features_train.append(features[temp_idx])
        elif args.task_name in ["chid"]:
            gd_idx = int(label)
            for j in range(7):
                temp_idx = 7 * idx_ + j
                if j == gd_idx:
                    features[temp_idx].label = 0  # entail
                else:
                    features[temp_idx].label = 1  # not_entail
                features_train.append(features[temp_idx])
    # all part
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_token_ids = torch.tensor([f.token_idx for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    # train part
    train_input_ids = torch.tensor([f.input_ids for f in features_train], dtype=torch.long)
    train_attention_mask = torch.tensor([f.attention_mask for f in features_train], dtype=torch.long)
    train_token_type_ids = torch.tensor([f.token_type_ids for f in features_train], dtype=torch.long)
    train_lens = torch.tensor([f.input_len for f in features_train], dtype=torch.long)
    train_token_ids = torch.tensor([f.token_idx for f in features_train], dtype=torch.long)
    if output_mode == "classification":
        train_labels = torch.tensor([f.label for f in features_train], dtype=torch.long)
    elif output_mode == "regression":
        train_labels = torch.tensor([f.label for f in features_train], dtype=torch.float)
    all_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels, all_token_ids)
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, train_lens, train_labels, train_token_ids)
    return all_dataset, test_sentences_labels, train_dataset


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
    elif data_type == 'boot':
        examples, test_sentences_labels = processor.get_unlabeled_examples(args.data_dir,task_label_description)
        num = args.cat_num * len(task_label_description)
        examples, test_sentences_labels = examples[:num], test_sentences_labels[:num]
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

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    # 训练ocnli或者cmnli的时候，会把classifier的权重也保存了，使用时不需要这部分权重
    # bin_file_path=os.path.join(args.model_name_or_path, WEIGHTS_NAME)

    # source_state_dict=torch.load(bin_file_path)
    # source_state_dict_keys=list(source_state_dict.keys())

    # for key in source_state_dict_keys:
    #     if key.startswith("classifier"):
    #         del source_state_dict[key]
    # torch.save(source_state_dict,bin_file_path)

    if args.bert_multi:
        model = BertMulti(args)
        print("***********using bert multi loss**************")
    else:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training

    if args.do_intermediate_train:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        _, _ = intermidate_train(args, train_dataset, model, tokenizer)
        logger.info("Will Saving model to %s", args.output_dir)

    if args.load_cmnli_model:
        model.load_state_dict(torch.load(os.path.join(args.cmnli_path, 'pytorch_model.bin'),
                                         map_location=torch.device('cpu')))
        logger.info("**load cmnli model***")

    if args.bootStrap:
        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        BootStrappingTraining(args, train_dataset, label_list, model, tokenizer)


    if args.do_train:
        train_dataset,_ = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if args.bert_multi:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model_{}.bin'.format(args.task_idx)))
        else:
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args_{}.bin'.format(args.task_idx)))



    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("%s = %s\n" % (key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        if args.bert_multi:
            model = BertMulti(args)
            model_path = os.path.join(args.output_dir, 'pytorch_model_{}.bin'.format(args.task_idx))
            logger.info("get model from: {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
            model.to(args.device)
            predict(args, model, tokenizer, label_list, prefix='')
            return
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            predict(args, model, tokenizer, label_list, prefix=prefix)


if __name__ == "__main__":
    main()
