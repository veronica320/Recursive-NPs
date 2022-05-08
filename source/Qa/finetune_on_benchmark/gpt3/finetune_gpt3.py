'''
Finetune GPT3 on the training set of existing benchmarks: MPE, ADEPT.
End-of-prompt separator:\n\n###\n\n
Start-of-completion separator:a whitespace
End token:\n, ### or any other special token which doesn't appear within any completion.
'''

import os
import sys
import argparse

os.chdir("../../../..")

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

data_dir = "data/existing_benchmarks"

assert args.model in ["ada", "babbage", "curie"]


train_dir = f"data/{args.data}/jsonl/train.jsonl"
os.system(f'nohup openai api fine_tunes.create \
			-t {train_dir} \
			-m {args.model} \
			--no_packing \
			source/Qa/finetune_on_benchmark/gpt3/logs/{args.data}_{args.model}.log 2>&1 &')