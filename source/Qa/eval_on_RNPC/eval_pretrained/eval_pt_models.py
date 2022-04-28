import os
import csv
import sys
from Qa.eval_on_RNPC.eval_pretrained.conditionalLM import PPLScorer
from Qa.eval_on_RNPC.eval_pretrained.MLM import MLMScorer

from Qa.utils import compute_scores, n_classes_dict_NP, read_dataset, subtask_fieldnames, unchanged_fields, query
from Qa.eval_on_RNPC.eval_pretrained.sanity_check import score_prompts, score_two_single_sents
import openai
import time
import argparse

def initialize_LM(model_name):
	if "gpt2" in model_name:
		LM = PPLScorer(model_name)
	else:
		LM = MLMScorer(model_name)

if __name__ == "__main__":

	root_dir = "/path/to/Recursive-NPs"
	os.chdir(root_dir)

	parser = argparse.ArgumentParser(description='Process finetune config.')
	parser.add_argument("--model_class",
	                    required=True,
	                    )
	parser.add_argument("--model_name",
	                    required=True,
	                    )
	parser.add_argument("--subtask_ids",
	                    required=True,
	                    help="[0,1,2]"
	                    )
	parser.add_argument("--trial",
	                    action="store_true"
	                    )

	args = parser.parse_args()
	model_class= args.model_class
	model_name = args.model_name
	subtask_ids = eval(args.subtask_ids)
	trial = args.trial

	openai.api_key = os.getenv("OPENAI_API_KEY")

	subtask_dir = f"data/RNPC/tasks"
	subtask_cands = ["SPTE", "MPTE", "EPC"]
	subtasks = [subtask for id, subtask in enumerate(subtask_cands) if id in subtask_ids]
	thresh = [0.1, 0.5, 1, 2, 3, 5][1]

	print(model_name)
	exceptions = 0

	if model_class == "gpt2":
		model = PPLScorer(model_name)
	else:
		model = None

	print(f"LM intialized: {model_name}.")

	for subtask in subtasks:

		print(f"Task: {subtask}")

		frn = f"{subtask_dir}/{subtask}.csv"
		fwn = f"output_dir/RNPC/eval_pretrained/{subtask}/{model_name}.csv"

		gold_labels, pred_labels = [], []
		with open(frn, 'r') as fr, open(fwn, 'w') as fw:
			reader = csv.DictReader(fr, delimiter = "\t")
			fieldnames = reader.fieldnames + ["pred", "sent1_score", "sent0_score"]
			writer = csv.DictWriter(fw, fieldnames=fieldnames, delimiter = "\t")
			writer.writeheader()

			for row_id, row in enumerate(reader):

				if row_id % 100 == 0:
					time.sleep(60) # avoid rate limit exceeded errors

				new_row = row.copy()

				if "TE" in subtask:
					premise, prompt0, prompt1, ending, gold_label = row["premise"], row["prompt0"], row["prompt1"], row["ending"], row["label"]
					prompt0 = f"{premise} {prompt0}"
					prompt1 = f"{premise} {prompt1}"

					score0, score1 = score_prompts(prompt0, prompt1, ending, model_name, model)

					if score0 < score1:
						new_row["pred"] = 0
					elif score0 > score1:
						new_row["pred"] = 1
					else:
						new_row["pred"] = gold_label

				else:
					sentence0, sentence1, gold_label = row["sentence0"], row["sentence1"], row["label"]
					score0, score1 = score_two_single_sents(sentence0, sentence1, model_name, model)
					abs_diff = abs(score1 - score0)
					if abs_diff < thresh:
						new_row["pred"] = 2
					else:
						if score1 < score0: # sentence 0 is less likely
							new_row["pred"] = 1
						else: # sentence 0 is more likely
							new_row["pred"] = 0

				new_row["sent1_score"], new_row["sent0_score"] = score1, score0

				writer.writerow(new_row)

				gold_labels.append(int(row["label"]))
				pred_labels.append(int(new_row["pred"]))

		compute_scores(n_classes_dict_NP[subtask], gold_labels, pred_labels)
		print("\n")


