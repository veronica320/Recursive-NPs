'''Given models finetuned on RNPC (200 examples), evaluate on RNPC tasks.'''

import os
import sys
os.chdir("../..")

root_dir = os.getcwd()
sys.path.append(f"{root_dir}/source")

# config
from configuration import Config
config_path = (f'source/Qb/config.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
import torch

from Qa.utils import compute_scores, label_id2text, n_classes_dict_NP, task_fieldnames, unchanged_fields
from utils import entailment, event_plausibility

import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification



if __name__ == "__main__":

	# config
	cache_dir = config.cache_dir
	task = eval(config.task)
	if task not in ["SPTE", "MPTE", "EPC"]:
		raise ValueError("Unspported task. Please choose from 'SPTE', 'MPTE', and 'EPC'.")

	n_train_examples = eval(config.n_train_examples)

	# optimal models for each task
	model_dict = {"SPTE": "roberta-large-mnli",
	              "MPTE": "MPE_roberta",
	              "EPC": "ADEPT_roberta-l"
	              }

	# model architecture
	model_arch = model_dict[task]
	print(f"Evaluating {model_arch} on {task}...")
	print(f"# Training examples: {n_train_examples}")

	# we shared the model with 200 examples on huggingface model hub
	# you can use our pretrained checkpoint
	if n_train_examples == 200:
		output_model_dir = "veronica320"
	# for other numbers of training examples, you should first finetune the model
	else:
		output_model_dir = "output_model_dir/inoculation"

	model_name = f"{output_model_dir}/{task}_{model_arch}_{n_train_examples}"

	tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
	model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(f"cuda:0")

	frn = f"data/RNPC/inoculation/{task}/test.tsv"
	model_name_simple = model_name.split("/")[-1]

	# model prediction directory
	pred_dir = f"output_dir/RNPC/inoculation/{task}"
	if not os.path.isdir(pred_dir):
		os.makedirs(pred_dir)
	fwn = f"{pred_dir}/{model_name_simple}.csv"

	gold_labels, pred_labels = [], []

	with open(frn, "r") as fr, open(fwn, "w") as fw:
		reader = csv.DictReader(fr, delimiter="\t")

		writer = csv.DictWriter(fw,fieldnames=task_fieldnames[task])
		writer.writeheader()

		for i, row in enumerate(reader):
			new_row = {}

			for field in unchanged_fields:
				if field in row:
					new_row[field] = row[field]

			# prediction
			if task in ["SPTE", "MPTE"]:
				premise, hypothesis, gold_label = row["sentence1"], row["sentence2"], row["label"]
				confidence, pred_label_id = entailment(model, tokenizer, model_name, premise, hypothesis)
			else:
				first_event, second_event, gold_label = row["sentence1"], row["sentence2"], row["label"]
				confidence, pred_label_id = event_plausibility(model, tokenizer, model_name, first_event, second_event)

			gold_label_id = int(row["label"])
			if gold_label_id == 2:
				gold_label_id = 1 # we use Entailment = 1 in RNPC, but previous benchmarks use Entailment = 2

			gold_label = label_id2text(gold_label_id, task)
			pred_label = label_id2text(pred_label_id, task)
			gold_labels.append(gold_label_id)
			pred_labels.append(pred_label_id)

			new_row["confidence"], new_row["gold label"], new_row["pred label"] = confidence, gold_label, pred_label
			writer.writerow(new_row)

	# compute accuracy, precision, recall, and f1
	scores = compute_scores(n_classes_dict_NP[task], gold_labels, pred_labels)
	print(scores)