## Convert RNPC examples to the format for perplexity testing.

import os
import csv
from utils import task_name_dict

def convert_row_to_ppl_example(row, subtask):

	pos_infix = " necessarily"
	neg_infix = "n't necessarily"
	if subtask in ["single_premise_TE", "multi_premise_TE"]:
		premise, hypothesis = row["premise"], row["hypothesis"]
		hyp_tokens = hypothesis.split()

		TE_prefix_tokens = hyp_tokens[:2]
		try:
			assert TE_prefix_tokens[-1] in ["is", "are", "am", "has", "have"]
		except:
			raise AssertionError(f"Unknown TE prefix: {TE_prefix_tokens}; row: {row}")
		TE_prefix = ' '.join(TE_prefix_tokens)

		TE_suffix_tokens = hyp_tokens[2:]
		TE_suffix = ' '.join(TE_suffix_tokens)

		if TE_prefix_tokens[-1] in ["is", "are", "am"]:
			neg_prefix = f"{TE_prefix}{neg_infix}" # e.g. "This is not necessarily"
			pos_prefix = f"{TE_prefix}{pos_infix}" # e.g. "This is necessarily"
		else:
			if TE_prefix_tokens[-1] == "has":
				aux = "does"
			elif TE_prefix_tokens[-1] == "have":
				aux = "do"
			else:
				raise ValueError(f"unknown prefix: {TE_prefix}")
			neg_prefix = f"{TE_prefix_tokens[0]} {aux} {neg_infix} have" # "He does not necessarily have"
			pos_prefix = f"{TE_prefix_tokens[0]} {pos_infix} have" # "He does necessarily have"

		neg_hypothesis = f"{neg_prefix} {TE_suffix}"

		pos_hypothesis = f"{pos_prefix} {TE_suffix}"

		if row["label"] == "entailment":
			label = 1
		else:
			label = 0

		new_row = {"id": row["id"],
		           "combo": row["combo"],
		           "source NP": row["source NP"],
		           "premise": row["premise"],
		           "prompt1": pos_prefix,
		           "prompt0": neg_prefix,
		           "ending": TE_suffix,
		           "label": label
		           }

	elif subtask == "event_plausibility":

		if row["label"] == "more_likely":
			label = 0
		elif row["label"] == "less_likely":
			label = 1
		else:
			label= 2
		new_row = {"id": row["id"],
		           "combo": row["combo"],
		           "source NP": row["source NP"],
		           "sentence1": row["first_event"],
		           "sentence0": row["second_event"],
		           "label": label
		           }
		return new_row


	else:
		raise ValueError(f"Unknown subtask: {subtask}.")

	return new_row

if __name__ == "__main__":

	root_dir = "/path/to/Recursive-NPs"
	os.chdir(root_dir)

	subtask_dir = "data/RNPC/"
	task_abbr = ["SPTE", "MPTE", "EPC"][2]
	subtask = task_name_dict[task_abbr]
	subtasks = ["single_premise_TE","multi_premise_TE", "event_plausibility"]
	batch = 4

	for subtask in subtasks:
		if "TE" in subtask:
			fieldnames = ["id", "combo", "source NP", "premise", "prompt1", "prompt0", "ending", "label"]
		else:
			fieldnames = ["id", "combo", "source NP", "sentence1", "sentence0", "label"]

		frn = f"{subtask_dir}/tasks/{subtask}.csv"

		output_dir = f"{subtask_dir}/tasks_PPL"
		fwn = f"{output_dir}/{subtask}.csv"
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)

		with open(frn, 'r') as old_fr, open(fwn, 'w') as fw:
			reader = csv.DictReader(old_fr, delimiter = "\t")
			writer = csv.DictWriter(fw, fieldnames=fieldnames, delimiter = "\t")
			writer.writeheader()

			for row in reader:
				new_row = convert_row_to_ppl_example(row, subtask)
				writer.writerow(new_row)






