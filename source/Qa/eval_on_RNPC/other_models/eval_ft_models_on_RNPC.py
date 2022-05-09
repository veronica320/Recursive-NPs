'''Given models finetuned on existing benchmarks, evaluate on RNPC tasks.'''

import os
import sys
os.chdir("../../../..")

root_dir = os.getcwd()
sys.path.append(f"{root_dir}/source")

# config
from configuration import Config
config_path = (f'source/Qa/eval_on_RNPC/other_models/config.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
import torch

from Qa.utils import compute_scores, n_classes_dict_NP, label_text2id, label_id2text, task_fieldnames, unchanged_fields
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

	# models available for evaluation for each task
	model_dict = {"SPTE": ["textattack/bert-base-uncased-snli",
                            "textattack/bert-base-uncased-MNLI",
                            "roberta-large-mnli",
							"facebook/bart-large-mnli",],
	              "MPTE": ["veronica320/MPE_bert",
                           "veronica320/MPE_bert-l",
                           "veronica320/MPE_roberta",
                           "veronica320/MPE_roberta-l"],
	              "EPC": ["veronica320/ADEPT_bert",
	                      "veronica320/ADEPT_bert-l",
	                      "veronica320/ADEPT_roberta",
	                      "veronica320/ADEPT_roberta-l"]
	              }
	
	model_names = model_dict[task]

	print(f"Evaluating models on {task}: ...")

	for model_name in model_names:
		model_name_simple = model_name.split("/")[-1]
		print(model_name_simple)
		
		# load model
		tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
		model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(f"cuda:0")
		
		# input and output
		frn = f"data/RNPC/tasks/{task}.csv"
		pred_dir = f"output_dir/RNPC/eval_models_ft_on_benchmark/{task}"
		fwn = f"{pred_dir}/{model_name_simple}.csv"
		if not os.path.isdir(pred_dir):
			os.makedirs(pred_dir)

		gold_labels, pred_labels = [], []
	
		with open(frn, "r") as fr, open(fwn, "w") as fw:
			reader = csv.DictReader(fr)

			writer = csv.DictWriter(fw,fieldnames=task_fieldnames[task])
			writer.writeheader()

			for i, row in enumerate(reader):
				new_row = {}

				for field in unchanged_fields:
					if field in row:
						new_row[field] = row[field]

				# prediction
				if task in ["SPTE", "MPTE"]:
					premise, hypothesis = row["premise"], row["hypothesis"]
					confidence, pred_label_id = entailment(model, tokenizer, model_name, premise, hypothesis)
				else:
					first_event, second_event = row["first_event"], row["second_event"]
					confidence, pred_label_id = event_plausibility(model, tokenizer, model_name, first_event, second_event)

				gold_label = row["label"]
				pred_label = label_id2text(pred_label_id, task)
				gold_labels.append(label_text2id(gold_label, task))
				pred_labels.append(pred_label_id)

				new_row["confidence"], new_row["gold label"], new_row["pred label"] = confidence, gold_label, pred_label
				writer.writerow(new_row)

		# compute accuracy, precision, recall, and f1
		scores = compute_scores(n_classes_dict_NP[task], gold_labels, pred_labels)
		print(scores)
