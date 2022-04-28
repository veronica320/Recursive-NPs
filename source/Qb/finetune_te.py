#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse

os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/transformers/cache/dir/'
os.chdir('/path/to/Recursive-NPs')

parser = argparse.ArgumentParser(description='Process finetune config.')
parser.add_argument("--cuda",
					default=None,
					type=str,
					required=True,
					help="The GPU indices to use.",
					)
parser.add_argument("--mode",
					default=None,
					type=str,
					required=True,
					help="train_eval | eval | transfer | test.",
					)
parser.add_argument("--target",
					default=None,
					type=str,
					required=True,
					help="Target data to tune on.",
					)
parser.add_argument("--batch",
					default=None,
					type=str,
					required=True,
					help="batch number",
					)
parser.add_argument("--strategy",
					default=0,
					type=str,
					required=None,
					help="Split strategy: random|disjoint.",
					)
parser.add_argument("--n_train",
					default=0,
					type=str,
					required=False,
					help="Number of finetuning examples for inoculation.",
					)
parser.add_argument("--task",
					default=None,
					type=str,
					required=True,
					help="Task format.",
					)
parser.add_argument("--model",
					default='bert',
					type=str,
					required=False,
					help="Model architecture to use.",
					)
parser.add_argument("--epochs",
					default='5',
					type=str,
					required=False,
					help="Number of epochs.",
					)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
import torch
import transformers

model_type = {
	'roberta': 'roberta',
	'roberta-l': 'roberta',
	'roberta-large-mnli': 'roberta',
	'bert': 'bert',
	'mbert': 'bert',
	'mbert-c': 'bert',
	'xlnet': 'xlnet',
	'xlm-roberta': 'xlm-roberta',
	'xlm-roberta-l': 'xlm-roberta',
	'xlm': 'xlm',
	'deberta': ''
}

model_name = {
	'roberta': 'roberta-base',
	'roberta-l': 'roberta-large',
	'roberta-large-mnli': 'roberta-large-mnli',
	'bert': 'bert-base-uncased',
	'bert-l': 'bert-large-uncased',
	'mbert': 'bert-base-multilingual-uncased',
	'mbert-c': 'bert-base-multilingual-cased',
	'xlnet': 'xlnet-base-cased',
	'xlm-roberta': 'xlm-roberta-base',
	'xlm-roberta-l': 'xlm-roberta-large',
	'xlm': 'xlm-mlm-100-1280',
}

if args.model in model_name:
	os.environ['MODEL_NAME'] = model_name[args.model]
else:
	os.environ['MODEL_NAME'] = f'output_model_dir/{args.model}'

os.environ['TRANS_DIR'] = 'transformers/examples/text-classification'
inoculation_dir = "output_model_dir/inoculation"
if not os.path.isdir(inoculation_dir):
	os.makedirs(inoculation_dir)
os.environ['OUT_DIR'] = f'{inoculation_dir}/{args.target}_{args.strategy}_{args.model}_{args.n_train}'

task_name_mapping = {"SPTE":"single_premise_TE",
                     "MPTE":"multi_premise_TE",
                     "EPC":"event_plausibility"}

if args.n_train == "all":
	wnli_dir = "data/NP_data/test_data/subtasks_wnli/"
	os.environ['DATA_DIR'] = f'{wnli_dir}/{task_name_mapping[args.target]}/batch{args.batch}'

else:
	wnli_dir = "data/NP_data/inoculation_data/subtasks_wnli"
	os.environ['DATA_DIR'] = f'{wnli_dir}/{task_name_mapping[args.target]}/batch{args.batch}/{args.strategy}'

	os.chdir(os.environ['DATA_DIR'])
	os.system(f"rm cached_train_*")
	os.system(f"rm cached_dev_*")
	os.system(f"cp train_{args.n_train}.tsv train.tsv" )
	os.chdir('/path/to/Recursive-NPs')


if args.mode == 'train_eval':
	os.system(f'python $TRANS_DIR/run_glue.py \
	  --model_name_or_path $MODEL_NAME \
	  --task_name {args.task} \
	  --do_train \
	  --do_eval \
	  --data_dir $DATA_DIR \
	  --per_device_train_batch_size 8 \
	  --per_device_eval_batch_size 24 \
	  --learning_rate 1e-5 \
	  --num_train_epochs {args.epochs} \
	  --max_seq_length 128 \
	  --output_dir $OUT_DIR \
	  --evaluation_strategy steps \
	  --logging_steps 20 \
	  --save_steps -1 \
	  --overwrite_output_dir \
	  --run_name {args.target}_{args.strategy}_{args.model}_{args.n_train}_train\
	  --fp16')


if args.mode == "train":
	os.system(f'python $TRANS_DIR/run_glue.py \
	  --model_name_or_path $MODEL_NAME \
	  --task_name {args.task} \
	  --do_train \
	  --data_dir $DATA_DIR \
	  --per_device_train_batch_size 8 \
	  --per_device_eval_batch_size 24 \
	  --learning_rate 1e-5 \
	  --num_train_epochs {args.epochs} \
	  --max_seq_length 128 \
	  --output_dir $OUT_DIR \
	  --evaluation_strategy steps \
	  --logging_steps 100 \
	  --save_steps -1 \
	  --overwrite_output_dir \
	  --run_name {args.target}_{args.strategy}_{args.model}_{args.n_train}_train\
	  --fp16')

if args.mode == 'eval':
	os.system(f'python $TRANS_DIR/run_glue.py \
	  --model_name_or_path $MODEL_NAME \
	  --task_name {args.task} \
	  --do_eval \
	  --data_dir $DATA_DIR \
	  --per_gpu_eval_batch_size 32 \
	  --max_seq_length 200 \
	  --output_dir $OUT_DIR \
	  --save_steps -1 \
	  --run_name {args.target}_{args.strategy}_{args.model}_eval\
	  --fp16')

if args.mode == 'test':
	os.system(f'python $TRANS_DIR/run_glue.py \
	  --model_name_or_path $OUT_DIR \
	  --task_name {args.task} \
	  --do_eval \
	  --data_dir $DATA_DIR \
	  --per_gpu_eval_batch_size 32 \
	  --max_seq_length 200 \
	  --output_dir $OUT_DIR \
	  --save_steps -1 \
	  --run_name {args.target}_{args.model}_test\
	  --fp16')