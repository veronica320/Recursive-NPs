import os
import numpy as np
import csv
cuda = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import task_name_dict
from Qa.utils import compute_scores, n_classes_dict_NP


def entailment(model_name, premise, hypothesis):
	global model

	# textattack models have class 1 as entailment
	if "textattack" in model_name:
		entail_idx = 1
	else:
		entail_idx = 2


	x = tokenizer.encode(premise, hypothesis, return_tensors='pt')
	logits = model(x.to("cuda:0"))[0]

	probs = logits.softmax(dim=1)[0]
	probs = probs.cpu().detach().numpy()

	entail_prob = probs[entail_idx]
	nonentail_prob = sum([prob for i,prob in enumerate(probs) if i != entail_idx])

	if entail_prob >= nonentail_prob:
		return entail_prob, 1  # entail
	else:
		return nonentail_prob, 0  # non-entail

def reformat_event(event):
	event = event.strip(".")
	event = event[0].lower() + event[1:]
	return event

def event_plausibility(model_name, first_event, second_event):
	global model

	x = tokenizer.encode(first_event, second_event, return_tensors='pt')
	logits = model(x.to("cuda:0"))[0]

	probs = logits.softmax(dim=1)[0]
	probs = probs.cpu().detach().numpy()

	pred_label = np.argmax(probs)
	confidence = probs[pred_label]

	return confidence, pred_label

def gold_label_to_numeric(gold_label, subtask):
	if subtask in ["single_premise_TE", "multi_premise_TE"]:
		assert gold_label in ["entailment", "non-entailment"]
		if gold_label == "entailment":
			return 1
		elif gold_label == "non-entailment":
			return 0

	elif subtask in ["event_plausibility"]:
		assert gold_label in ["more_likely", "equally_likely", "less_likely"]
		if gold_label == "more_likely":
			return 2
		elif gold_label == "equally_likely":
			return 1
		elif gold_label == "less_likely":
			return 0

	else:
		print(subtask)


if __name__ == "__main__":

	root_dir = "/path/to/Recursive-NPs"
	os.chdir(root_dir)

	cache_dir = '/transformers/cache/dir/'

	# models
	model_dict = {"single_premise_TE": ["facebook/bart-large-mnli",
	                                    "roberta-large-mnli",
	                                    "joeddav/xlm-roberta-large-xnli",
	                                    "textattack/bert-base-uncased-snli",
	                                    "textattack/bert-base-uncased-MNLI"],
	              "multi_premise_TE": ["output_model_dir/MPE_bert",
	                                   "output_model_dir/MPE_bert-l",
	                                   "output_model_dir/MPE_roberta",
	                                   "output_model_dir/MPE_roberta-l"],
	              "event_plausibility": ["output_model_dir/ADEPT_bert",
	                                     "output_model_dir/ADEPT_bert-l",
	                                     "output_model_dir/ADEPT_roberta",
	                                     "output_model_dir/ADEPT_roberta-l",
	                                     ]
	              }

	# config
	task_abbr = ["SPTE", "MPTE", "EPC"][2]
	subtask = task_name_dict[task_abbr]

	print(subtask)

	model_names = model_dict[subtask][-1:]

	for model_name in model_names:
		print(model_name)

		tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
		model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(f"cuda:0")
		frn = f"data/RNPC/tasks/{task_abbr}.csv"

		model_name_simple = model_name.split("/")[-1]

		pred_dir = f"output_dir/RNPC/{task_abbr}"
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
