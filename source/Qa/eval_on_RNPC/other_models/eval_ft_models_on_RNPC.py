'''Evaluates finetuned models on RNPC tasks.'''

import os
from configuration import Config

# config
os.chdir("../../../..")
config_path = (f'source/Qa/eval_on_RNPC/other_models/config.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
import torch
root_dir = config.root_dir

import sys
sys.path.append(f"{root_dir}/source")
from Qa.utils import compute_scores, n_classes_dict_NP, label_text2id, label_id2text, task_fieldnames, unchanged_fields

import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import task_name_dict


def entailment(model_name, premise, hypothesis):
	'''Predicts whether the premise entails (1 yes, 0 no) the hypothesis.'''

	global model
	global tokenizer

	# textattack models have class 1 as entailment
	if "textattack" in model_name:
		entail_idx = 1
	# other models have class 2 as entailment
	else:
		entail_idx = 2

	x = tokenizer.encode(premise, hypothesis, return_tensors='pt')
	logits = model(x.to("cuda:0"))[0]

	probs = logits.softmax(dim=1)[0]
	probs = probs.cpu().detach().numpy()

	entail_prob = probs[entail_idx]
	# summing up the probs for contradiction and neutral, as the prob for non-entailment
	nonentail_prob = sum([prob for i,prob in enumerate(probs) if i != entail_idx])

	if entail_prob >= nonentail_prob:
		return entail_prob, 1  # entail
	else:
		return nonentail_prob, 0  # non-entail

def event_plausibility(model_name, first_event, second_event):
	'''Predicts whether second_event is more (2)/ equally (1)/ less likely (0) than first_event.'''

	global model
	global tokenizer

	x = tokenizer.encode(first_event, second_event, return_tensors='pt')
	logits = model(x.to("cuda:0"))[0]

	probs = logits.softmax(dim=1)[0]
	probs = probs.cpu().detach().numpy()

	pred_label = np.argmax(probs)
	confidence = probs[pred_label]

	return confidence, pred_label



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
	              "MPTE": ["output_model_dir/MPE_bert",
                           "output_model_dir/MPE_bert-l",
                           "output_model_dir/MPE_roberta",
                           "output_model_dir/MPE_roberta-l"],
	              "EPC": ["output_model_dir/ADEPT_bert",
	                      "output_model_dir/ADEPT_bert-l",
	                      "output_model_dir/ADEPT_roberta",
	                      "output_model_dir/ADEPT_roberta-l"]
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
					confidence, pred_label_id = entailment(model_name, premise, hypothesis)
				else:
					first_event, second_event = row["first_event"], row["second_event"]
					confidence, pred_label_id = event_plausibility(model_name, first_event, second_event)

				gold_label = row["label"]
				pred_label = label_id2text(pred_label_id, task)
				gold_labels.append(label_text2id(gold_label, task))
				pred_labels.append(pred_label_id)

				new_row["confidence"], new_row["gold label"], new_row["pred label"] = confidence, gold_label, pred_label
				writer.writerow(new_row)

		# compute accuracy, precision, recall, and f1
		scores = compute_scores(n_classes_dict_NP[task], gold_labels, pred_labels)
		print(scores)
