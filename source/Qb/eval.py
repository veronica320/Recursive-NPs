import os
import numpy as np
import csv
cuda = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from Qa.eval_on_RNPC.eval_finetuned.other_models.eval_ft_models_on_RNPC import  entailment, reformat_event, event_plausibility, gold_label_to_numeric
from Qa.utils import compute_scores, n_classes_dict_NP


if __name__ == "__main__":

	root_dir = "/path/to/Recursive-NPs"
	os.chdir(root_dir)

	cache_dir = '/transformers/cache/dir/'

	# models
	subtask_shorthands = ["SPTE","MPTE","EPC"]
	task_name_mapping = {"SPTE": "single_premise_TE",
	                     "MPTE": "multi_premise_TE",
	                     "EPC": "event_plausibility"}

	# config
	subtask_shorthand = subtask_shorthands[1]
	range_of_train_examples = [0, 10, 25, 50, 75, 100, 125, 150, 175, 200]
	model_dict = {"single_premise_TE": "roberta-large-mnli",
	              "multi_premise_TE": ["MPE_bert", "MPE_bert-l", "MPE_roberta", "MPE_roberta-l"][2],
	              "event_plausibility": "ADEPT_roberta-l",
	              }


	subtask = task_name_mapping[subtask_shorthand]
	model_arch = model_dict[subtask]
	print(subtask, model_arch)

	for n_train_examples in range_of_train_examples:
		print("\n")
		print(n_train_examples)
		if n_train_examples == 0:
			if model_arch == "roberta-large-mnli":
				model_name = model_arch
			else:
				output_model_dir = "output_model_dir"
				model_name = f"{output_model_dir}/{model_arch}"
		else:
			output_model_dir = "output_model_dir/inoculation"
			model_name = f"{output_model_dir}/{subtask_shorthand}_{model_arch}_{n_train_examples}"


		tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
		model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(f"cuda:0")

		frn = f"data/RNPC/inoculation/{subtask}/test.csv"
		model_name_simple = model_name.split("/")[-1]
		pred_dir = f"output_dir/RNPC/inoculation/{subtask}"

		fwn = f"{pred_dir}/{model_name_simple}.csv"
		if not os.path.isdir(pred_dir):
			os.makedirs(pred_dir)

		gold_labels, pred_labels = [], []

		subtask_fieldnames = {
			"single_premise_TE": ["id", "combo", "source NP", "premise", "hypothesis", "gold label","pred label", "confidence"],
			"multi_premise_TE": ["id", "combo", "source NP", "premise", "hypothesis", "gold label","pred label", "confidence"],
			"event_plausibility": ["id", "combo", "source NP", "first_event", "second_event", "gold label","pred label", "confidence"]
			}

		with open(frn, "r") as fr, open(fwn, "w") as fw:
			reader = csv.DictReader(fr, delimiter="\t")

			writer = csv.DictWriter(fw,fieldnames=subtask_fieldnames[subtask])
			writer.writeheader()

			unchanged_fields = ["id","combo","source NP", "premise", "hypothesis", "first_event", "second_event"]

			for i, row in enumerate(reader):
				new_row = {}

				for field in unchanged_fields:
					if field in row:
						new_row[field] = row[field]
				new_row["gold label"] = gold_label_to_numeric(row["label"], subtask)

				if subtask in ["single_premise_TE", "multi_premise_TE"]:
					premise, hypothesis, gold_label = row["premise"], row["hypothesis"], row["label"]
					gold_label = gold_label_to_numeric(gold_label, subtask)
					confidence, pred_label = entailment(model_name, premise, hypothesis)

				else:
					first_event, second_event, gold_label = row["first_event"], row["second_event"], row["label"]
					gold_label = gold_label_to_numeric(gold_label, subtask)
					confidence, pred_label = event_plausibility(model_name, first_event, second_event)

				new_row["confidence"], new_row["pred label"] = confidence, pred_label
				writer.writerow(new_row)
				gold_labels.append(gold_label)
				pred_labels.append(pred_label)

		compute_scores(n_classes_dict_NP[subtask], gold_labels, pred_labels)