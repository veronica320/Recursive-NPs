import csv
# import openai
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import openai

task_fieldnames = {
	"SPTE": ["id", "combo", "source NP", "premise", "hypothesis", "gold label", "pred label", "confidence"],
	"MPTE": ["id", "combo", "source NP", "premise", "hypothesis", "gold label", "pred label", "confidence"],
	"EPC": ["id", "combo", "source NP", "first_event", "second_event", "gold label", "pred label", "confidence"]
}

## number of classes in each RNPC task
n_classes_dict_NP = {
	"SPTE":2,
	"MPTE":2,
	"EPC":3,
}

## number of classes in existing benchmark(s) corresponding to each RNPC task
n_classes_dict_benchmark = {
	"SPTE":3,
	"MPTE":3,
	"EPC":3,
}

# fields that we can directly copy from frn to fwn
unchanged_fields = ["id", "combo", "source NP", "premise", "hypothesis", "first_event", "second_event"]

# label text-index mapping
def label_text2id(label, task):
	'''Converts a label from text to index.'''

	if task in ["SPTE", "MPTE"]:
		assert label in ["entailment", "non-entailment"]
		if label == "entailment":
			return 1
		elif label == "non-entailment":
			return 0
	elif task in ["EPC"]:
		assert label in ["more_likely", "equally_likely", "less_likely"]
		if label == "more_likely":
			return 2
		elif label == "equally_likely":
			return 1
		elif label == "less_likely":
			return 0

def label_id2text(label, task):
	'''Converts a label from index to text.'''

	if task in ["SPTE", "MPTE"]:
		assert label in [0, 1]
		if label == 1:
			return "entailment"
		elif label == 0:
			return "non-entailment"
	elif task in ["EPC"]:
		assert label in [2, 1, 0]
		if label == 2:
			return "more_likely"
		elif label == 1:
			return "equally_likely"
		elif label == 0:
			return "less_likely"


## General util functions
def compute_scores(n_classes, gold_labels, pred_labels):
	if n_classes > 2:
		p = round(precision_score(gold_labels, pred_labels, average="weighted"), 3)
		r = round(recall_score(gold_labels, pred_labels, average="weighted"), 3)
		f = round(f1_score(gold_labels, pred_labels, average="weighted"), 3)
	else:
		p = round(precision_score(gold_labels, pred_labels), 3)
		r = round(recall_score(gold_labels, pred_labels), 3)
		f = round(f1_score(gold_labels, pred_labels), 3)
	weighted_f = round(f1_score(gold_labels, pred_labels, average="weighted"), 3)
	acc = round(accuracy_score(gold_labels, pred_labels), 3)
	scores = {"accuracy": acc,
	          "precision": p,
	          "recall": r,
	          "f1": f,
	          "weighted-F1": weighted_f
	          }
	return scores

## Util functions for GPT3
def read_dataset(frn):
	label_class = []
	rows = []
	with open(frn, 'r') as fr:
		reader = csv.DictReader(fr)
		for row in reader:
			rows.append(row)
			if row["label"] not in label_class:
				label_class.append(row["label"])
	dataset = {"examples": rows,
	           "label_classes": label_class
	           }
	return dataset

def add_missing_label(target_labels, label_probs):
	for label in target_labels:
		if label not in label_probs:
			label_probs[label] = 0.0
	return label_probs

def query(model, prompt):
	'''Query GPT3 model for prompt-based generation.'''
	response = openai.Completion.create(
		model=model,
		prompt=prompt,
		temperature=0,
		max_tokens=1,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0,
		logprobs=5,
	)
	if response["choices"][0]["text"] == "":
		raise ValueError(f"Null response: {response}\nPrompt: {[prompt]}")
	response_label, top_logprobs = int(response["choices"][0]["text"].strip()), \
	                               response["choices"][0]["logprobs"]["top_logprobs"][0]

	label_probs = {key:np.exp(value) for key, value in top_logprobs.items()}
	label_probs = add_missing_label({" 0"," 1"," 2"}, label_probs)
	return response_label, label_probs