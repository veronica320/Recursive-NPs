'''
Finetune GPT3 on the training set of existing benchmarks: MPE, ADEPT.
!!!NOTE!!!: at the time of our project, finetuning GPT-3 was free. Since it's NO LONGER the case now, we recommend that you try with a sample of the data first!
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
parser.add_argument("--model",
					type=str,
					required=True,
					help="Model architecture to use.",
					)
parser.add_argument("--trial",
                    type=bool,
                    action='store_true',
                    help="Trial run on a sample of the dataset.",
                    )

args = parser.parse_args()

data_dir = "data/existing_benchmarks"
log_dir = "source/Qa/finetune_on_benchmark/gpt3/logs/"
if not os.path.isdir(log_dir):
	os.makedirs(log_dir)

assert args.model in ["ada", "babbage", "curie"]

if args.trial:
	train_dir = f"data/{args.data}/jsonl/train_sample.jsonl"
else:
	train_dir = f"data/{args.data}/jsonl/train.jsonl"

os.system(f'nohup openai api fine_tunes.create \
			-t {train_dir} \
			-m {args.model} \
			--no_packing \
			source/Qa/finetune_on_benchmark/gpt3/logs/{args.data}_{args.model}.log 2>&1 &')