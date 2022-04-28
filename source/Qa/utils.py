import csv
import openai
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

subtask_fieldnames = {
	"single_premise_TE": ["id", "combo", "source NP", "premise", "hypothesis", "gold label", "pred label", "confidence"],
	"multi_premise_TE": ["id", "combo", "source NP", "premise", "hypothesis", "gold label", "pred label", "confidence"],
	"event_plausibility": ["id", "combo", "source NP", "first_event", "second_event", "gold label", "pred label", "confidence"]
}

## number of classes in NP test set
n_classes_dict_NP = {
	"single_premise_TE":2,
	"multi_premise_TE":2,
	"event_plausibility":3,
}


unchanged_fields = ["id", "combo", "source NP", "premise", "hypothesis", "first_event", "second_event"]

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
	print("ACC/P/R/F/Weighted-F1")
	print(acc, p, r, f, weighted_f)


## Util functions for GPT3
def read_dataset(frn):
	label_class = []
	rows = []
	with open(frn, 'r') as fr:
		reader = csv.DictReader(fr, delimiter = "\t")
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

def query(model, prompt, get_label_probs=False):
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
	if get_label_probs:
		response_label, top_logprobs = int(response["choices"][0]["text"].strip()), \
		                               response["choices"][0]["logprobs"]["top_logprobs"][0]

		label_probs = {key:np.exp(value) for key, value in top_logprobs.items()}
		label_probs = add_missing_label({" 0", " 1", " 2"}, label_probs)
		return response_label, label_probs

	else:
		response_label, top_logprobs = response["choices"][0]["text"].strip(), \
		                               response["choices"][0]["logprobs"]["top_logprobs"][0]
		return response_label, top_logprobs