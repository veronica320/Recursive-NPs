'''
Evaluate RNPC-based models on the harm detection test set.
'''

import os
import sys
os.chdir("../..")

root_dir = os.getcwd()
sys.path.append(f"{root_dir}/source")

# config
from configuration import Config
config_path = (f'source/Qd/config.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import csv

from source.Qd.model import HarmDetector
from source.Qd.utils import gold_label2idx, pred_label2idx
from source.Qa.utils import compute_scores


def gold_to_pred_label(label):
	assert label in ["GOOD", "HARM"]
	if label == "GOOD":
		return "unharmful"
	else:
		return "harmful"

def pred_to_gold_label(label):
	assert label in ["harmful", "unharmful"]
	if label == "harmful":
		return "HARM"
	else:
		return "GOOD"

if __name__ == "__main__":

	# config
	RNPC_task = eval(config.RNPC_task)

	print(f"Using {RNPC_task} model...")

	predictor = HarmDetector(config)

	data_dir = "data/harm_detection"
	frn = f"{data_dir}/test.csv"
	output_dir = "output_dir/harm_detection"
	fwn = f"{output_dir}/{RNPC_task}_pred.csv"

	total, correct = 0, 0

	with open(frn, 'r') as fr, open(fwn, 'w') as fw:
		reader = csv.DictReader(fr)
		writer = csv.DictWriter(fw, fieldnames=reader.fieldnames + ["pred label", "confidence", "provenance"])
		writer.writeheader()

		gold_labels, pred_labels = [], []

		for row in reader:

			query = row["Query"]
			gold_label_id = gold_label2idx(row["Label"])

			output = predictor.predict(query)

			pred = list(output.items())[0]
			pred_label = pred[0]

			row["pred label"] = pred_to_gold_label(pred_label)
			pred_label_id = pred_label2idx(pred_label)

			for col in ["confidence", "provenance"]:
				row[col] = output[pred_label][col]
			writer.writerow(row)

			gold_labels.append(gold_label_id)
			pred_labels.append(pred_label_id)

	scores = compute_scores(2, gold_labels, pred_labels)
	print(scores)