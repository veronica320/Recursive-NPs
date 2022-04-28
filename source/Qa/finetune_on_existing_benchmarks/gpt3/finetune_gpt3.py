#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse

os.chdir('/path/to/Recursive-NPs')

parser = argparse.ArgumentParser(description='Process finetune config.')
parser.add_argument("--data",
					default=None,
					type=str,
					required=True,
					help="Target data to tune on.",
					)
parser.add_argument("--format",
					default="",
					type=str,
					required=False,
					help="Format of data to tune on.",
					)
parser.add_argument("--model",
					type=str,
					required=True,
					help="Model architecture to use.",
					)

args = parser.parse_args()

fn2id = {"data/MPE/jsonl/train.jsonl": "file-mzUJ58Cu0rQk3r7eM4Gc4X3c",
         "data/ADEPT/jsonl/train.jsonl": "file-PicteiSiMnV1PACcjj22Yvwh",
         }

assert args.model in ["ada", "babbage", "curie"]

def convert_fn_to_id(fn):
	if fn in fn2id:
		return fn2id[fn]
	else:
		return fn


if args.format != "":
	train_dir = f"data/{args.data}/jsonl/{args.format}/train.jsonl"
	train_dir = convert_fn_to_id(train_dir)
	os.system(f'nohup openai api fine_tunes.create -t {train_dir} -m {args.model} --no_packing \
	> source/Qa/finetune_on_existing_benchmarks/gpt3/{args.data}_{args.format}_{args.model}.log 2>&1 &')
else:
	train_dir = f"data/{args.data}/jsonl/train.jsonl"
	train_dir = convert_fn_to_id(train_dir)
	os.system(f'nohup openai api fine_tunes.create -t {train_dir} -m {args.model} --no_packing \
	> source/Qa/finetune_on_existing_benchmarks/gpt3/{args.data}_{args.model}.log 2>&1 &')