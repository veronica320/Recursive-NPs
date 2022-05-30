import csv
import os
from source.transfer.model import HarmDetector
import ipdb
from source.eval_baseline.other_baselines.eval_baselines import compute_scores

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

def gold_label2idx(label):
	assert label in ["GOOD", "HARM"]
	if label == "GOOD":
		return 0
	else:
		return 1

def pred_label2idx(label):
	assert label in ["harmful", "unharmful"]
	if label == "harmful":
		return 1
	else:
		return 0


if __name__ == "__main__":
	root_dir = "/nlp/data/lyuqing-zharry/UG"
	os.chdir(root_dir)

	# config
	RNPC_task = ["SPTE", "EPC"][1]
	filter = [None, ["tattoo"]][1]
	print(f"Using {RNPC_task} model...")

	predictor = HarmDetector(RNPC_task)
	data_dir = "data/harm_detection/cleaned"
	frn = f"{data_dir}/all.csv"
	fwn = f"{data_dir}/{RNPC_task}_{filter}_pred.csv"

	total, correct = 0, 0

	with open(frn, 'r') as fr, open(fwn, 'w') as fw:
		reader = csv.DictReader(fr)
		writer = csv.DictWriter(fw, fieldnames=reader.fieldnames + ["pred label", "confidence", "provenance", "probs"])
		writer.writeheader()

		gold_labels, pred_labels = [], []

		for row in reader:

			query = row["Title"]
			gold_label_id = gold_label2idx(row["Label"])

			skip_query = False
			for word_to_filter in filter:
				if word_to_filter in query:
					skip_query = True
					break
			if skip_query:
				continue

			output = predictor.predict(query)
			if len(output.items()) > 1:
				print("Multiple NPs: ", output.items(), row)

			elif len(output.items()) == 0:
				print("No output: ", row)
				continue

			pred = list(output.items())[0]
			pred_label = pred[0]

			row["pred label"] = pred_to_gold_label(pred_label)
			pred_label_id = pred_label2idx(pred_label)


			for name in ["confidence", "provenance", "probs"]:
				row[name] = output[pred_label][name]
			writer.writerow(row)

			gold_labels.append(gold_label_id)
			pred_labels.append(pred_label_id)

	compute_scores(2, gold_labels, pred_labels)