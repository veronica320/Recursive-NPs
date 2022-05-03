'''
Evaluated finetuned GPT3 on the test set of previous benchmarks.
End-of-prompt separator:\n\n###\n\n
Before-completion separator:a whitespace
End token:\n, ### or any other special token which doesn't appear within any completion. (Then include as a stop sequence.)
'''
import csv
import json
import os

import numpy as np
import openai

from Qa.utils import compute_scores

finetuned_model_dict = {
	"MPTE":{
		"ada":"ada:ft-ccb-lab-members-2021-08-12-01-09-11",
		"curie":"curie:ft-ccb-lab-members-2021-08-12-02-24-03"
	},
	"EPC":{
		"ada":"ada:ft-ccb-lab-members-2021-08-13-00-40-39",
		"curie":"curie:ft-ccb-lab-members-2021-08-13-02-39-17",
	}
}


subtask_fieldnames = {
	"single_premise_TE": ["premise", "hypothesis", "gold label", "pred label", "confidence"],
	"multi_premise_TE": ["premises", "hypothesis", "gold label", "pred label", "confidence"],
	"event_plausibility": ["first_event", "second_event", "gold label", "pred label", "confidence"]
}

def query(model, prompt, n_classes):
	response = openai.Completion.create(
		model=model,
		prompt=prompt,
		temperature=0,
		max_tokens=1,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0,
		logprobs=n_classes,
	)
	if response["choices"][0]["text"] == "":
		raise ValueError(f"Null response: {response}\nPrompt: {[prompt]}")
	response_label, top_logprobs = response["choices"][0]["text"].strip(), response["choices"][0]["logprobs"]["top_logprobs"][0]
	label_probs_sum = sum([np.exp(value) for key, value in top_logprobs.items()])
	label_prob = np.exp(top_logprobs[response["choices"][0]["text"]]) / label_probs_sum

	return response_label, label_prob

if __name__ == "__main__":

	## config
	root_dir = "/path/to/Recursive-NPs"
	os.chdir(root_dir)
	task = ["single_premise_TE", "multi_premise_TE", "event_plausibility"][2]
	model = ["ada", "babbage", "curie"][2]

	gold_labels, pred_labels = [], []

	# For SPTE, the finetuning datasets (MNLI, SNLI) are too large.
	# OpenAI doesn't allow datasets of such sizes at the time of the work.

	# Evaluate on MPE
	if task == "multi_premise_TE":
		jsonl_dir = f"data/MPE/jsonl/"
		frn = f"{jsonl_dir}/test.jsonl"

		output_dir = f"data/MPE/jsonl/ft_gpt3_pred"
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)
		fwn = f"{output_dir}/{model}_test.tsv"

		with open(frn, 'r') as fr, open(fwn, 'w') as fw:
			writer = csv.DictWriter(fw, delimiter="\t", fieldnames=subtask_fieldnames[task])
			writer.writeheader()

			for row_id, row_obj in enumerate(fr.readlines()):
				if row_id % 50 == 0:
					print(f"{row_id} examples finished.")

				row = json.loads(row_obj)
				prompt, completion = row["prompt"], row["completion"]
				pred_label, pred_prob = query(finetuned_model_dict[task][model], prompt, n_classes_dict_benchmark[task])
				pred_label = int(pred_label.strip())

				gold_label = int(completion.strip())

				gold_labels.append(gold_label)
				pred_labels.append(pred_label)

				cols = prompt.split("\n")
				premises, hypothesis = cols[0], cols[1]
				premises = premises[len("Premises: "):]
				hypothesis = hypothesis[len("Hypothesis: "):]

				new_row = {}
				new_row["premises"] = premises
				new_row["hypothesis"] = hypothesis
				new_row["gold label"] = gold_label
				new_row["pred label"] = pred_label
				new_row["confidence"] = pred_prob
				writer.writerow(new_row)

	# Evaluate on ADEPT
	elif task == "event_plausibility":
		jsonl_dir = f"data/ADEPT/jsonl/"
		frn = f"{jsonl_dir}/test.jsonl"

		output_dir = f"data/ADEPT/jsonl/ft_gpt3_pred"
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)
		fwn = f"{output_dir}/{model}_test.tsv"

		with open(frn, 'r') as fr, open(fwn, 'a') as fw:
			writer = csv.DictWriter(fw, delimiter="\t", fieldnames=subtask_fieldnames[task])
			writer.writeheader()

			for row_id, row_obj in enumerate(fr.readlines()):

				if row_id % 50 == 0:
					print(f"{row_id} examples finished.")

				row = json.loads(row_obj)
				prompt, completion = row["prompt"], row["completion"]
				pred_label, pred_prob = query(finetuned_model_dict[task][model], prompt, n_classes_dict_benchmark[task])
				pred_label = int(pred_label.strip())

				gold_label = int(completion.strip())

				gold_labels.append(gold_label)
				pred_labels.append(pred_label)

				cols = prompt.split("\n")
				event1, event2 = cols[0], cols[1]
				event1 = event1[len("Event 1: "):]
				event2 = event2[len("Event 2: "):]

				new_row = {}
				new_row["first_event"] = event1
				new_row["second_event"] = event2
				new_row["gold label"] = gold_label
				new_row["pred label"] = pred_label
				new_row["confidence"] = pred_prob
				writer.writerow(new_row)

	compute_scores(n_classes_dict_benchmark[task], gold_labels, pred_labels)
