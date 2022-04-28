import csv
import os

import numpy as np
import openai

from utils import task_name_dict

from Qa.utils import compute_scores, read_dataset, subtask_fieldnames, unchanged_fields, query
from Qa.finetune_on_existing_benchmarks.gpt3.eval_gpt3_indomain import finetuned_model_dict

## number of classes in NP test set
n_classes_dict_NP = {
	"single_premise_TE":2,
	"multi_premise_TE":2,
	"event_plausibility":3,
}

def label2idx(task, label):
	if task in ["single_premise_TE", "multi_premise_TE"]:
		if label == "entailment":
			return 1
		elif label == "non-entailment":
			return 0
		else:
			raise ValueError(f"Unrecognized label: {label}")

	elif task == "event_plausibility":
		if label == "less_likely":
			return 0
		elif label == "equally_likely":
			return 1
		elif label == "more_likely":
			return 2


def map_class(task, pred_label, label_probs):

	if task == "multi_premise_TE":
		assert pred_label in [0, 1, 2]
		entail_prob = label_probs[" 2"]
		nonentail_prob = label_probs[" 0"] + label_probs[" 1"]
		total_prob = entail_prob + nonentail_prob
		if entail_prob >= nonentail_prob:
			return 1, entail_prob/total_prob # entail
		else:
			return 0, nonentail_prob/total_prob # non-entail

	elif task == "event_plausibility":
		assert pred_label in [0, 1, 2]
		total_prob = label_probs[" 0"] + label_probs[" 1"] + label_probs[" 2"]
		pred_prob = label_probs[f" {pred_label}"] / total_prob
		return pred_label, pred_prob

	else:
		raise ValueError(f"Unrecognized task: {task}")

def convert_example_to_prompt(task, example, format):
	prompt = ""
	if task == "multi_premise_TE":
		if format == "concat":
			premise, hypothesis = example["premise"], example["hypothesis"]
			prompt = f"Premises: {premise}\nHypothesis: {hypothesis}\n\n###\n\n"

		elif format == "sep":
			premise, hypothesis = example["premise"], example["hypothesis"]
			premise1, premise2 = premise.split(". ")
			premise1 += "."

	elif task == "event_plausibility":
			event1, event2 = example["first_event"], example["second_event"]
			prompt = f"Event 1: {event1}\nEvent 2: {event2}\n\n###\n\n"

	if prompt == "":
		raise ValueError("Empty prompt.")

	return prompt

if __name__ == "__main__":

	## config
	root_dir = "/path/to/Recursive-NPs"
	os.chdir(root_dir)
	task_abbr = ["SPTE", "MPTE", "EPC"]
	task = task_name_dict[task_abbr]
	model = ["ada", "curie"][1]

	frn = f"data/RNPC/tasks/{task_abbr}.csv"
	test_set = read_dataset(frn)

	output_dir = f"output_dir/RNPC/{task_abbr}/gpt3"
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	fwn = f"{output_dir}/{model}_finetuned.tsv"

	gold_labels, pred_labels = [], []

	with open(fwn, 'a') as fw:
		writer = csv.DictWriter(fw, delimiter="\t", fieldnames=subtask_fieldnames[task])
		writer.writeheader()

		for test_id, test_example in enumerate(test_set["examples"]):
			print(test_id)
			if test_id % 50 == 0:
				print(f"{test_id} examples finished.")

			prompt = convert_example_to_prompt(task, test_example, format)
			if format:
				raw_pred_label, raw_label_probs = query(finetuned_model_dict[task][format][model], prompt)
			else:
				raw_pred_label, raw_label_probs = query(finetuned_model_dict[task][model], prompt)

			pred_label, label_prob = map_class(task, raw_pred_label, raw_label_probs)
			if pred_label == None:
				print(prompt)
				print(raw_pred_label, raw_label_probs)

			gold_label = label2idx(task, test_example["label"])

			gold_labels.append(gold_label)
			pred_labels.append(pred_label)

			row = {}
			for field in unchanged_fields:
				if field in test_example:
					row[field] = test_example[field]
			row["gold label"] = gold_label
			row["pred label"] = pred_label
			row["confidence"] = label_prob
			writer.writerow(row)
			fw.flush()

	compute_scores(n_classes_dict_NP[task], gold_labels, pred_labels)



