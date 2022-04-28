import os
from Qa.eval_on_RNPC.eval_pretrained.conditionalLM import PPLScorer
from Qa.eval_on_RNPC.eval_pretrained.MLM import MLMScorer
from itertools import product
import csv
from Qa.utils import compute_scores
import openai
from bisect import bisect_left

def read_data(frn):
	with open(frn, 'r') as fr:
		reader = csv.DictReader(fr)
		rows = [row for row in reader]
		fieldnames = reader.fieldnames
		return rows, fieldnames

def write_preds(fwn, rows, fieldnames):
	with open(fwn, 'w') as fw:
		writer = csv.DictWriter(fw, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def align_length(prompt, gpt3_char_offset):
	n_prompt_chars = len(prompt)
	ending_start_token_idx = bisect_left(gpt3_char_offset, n_prompt_chars)
	return ending_start_token_idx

def score_ending_gpt3(prompt, ending, model_name):
	if prompt == "":
		sentence = ending
	else:
		sentence = f"{prompt} {ending}"
	response = openai.Completion.create(
		engine=model_name[5:],
		prompt=sentence,
		temperature=1,
		max_tokens=0,
		top_p=1,
		logprobs=1,
		echo=True,
		frequency_penalty=0,
		presence_penalty=0
	)
	logprobs = response["choices"][0]["logprobs"]

	if prompt == "":
		ending_start_token_idx = 1
		ending_length = len(logprobs["text_offset"]) - 1
	else:
		ending_start_token_idx = align_length(prompt, logprobs["text_offset"])
		ending_length = len(logprobs["text_offset"]) - ending_start_token_idx + 1
	if ending_start_token_idx == -1:
		raise ValueError(f"Alignment failed. {response}")
	neg_ll = -sum(logprobs["token_logprobs"][ending_start_token_idx:])
	avg_neg_ll = neg_ll / ending_length
	return avg_neg_ll



def mask(prompt, ending, model_name, model):
	mask_token_dict = {"bert": "[MASK]",
	                   "roberta": "<mask>"
	                   }
	mask_multiplier_dict = {"bert":{"n't": 3,
	                                "not": 2},
	                        "roberta": {"n't": 2,
	                                    "not": 2}
	                        }
	if "roberta" in model_name:
		model_type = "roberta"
	elif "bert" in model_name:
		model_type = "bert"
	else:
		return ValueError(f"Unknown model name: {model_name}")

	mask_token = mask_token_dict[model_type]
	orig_sentence = f"{prompt} {ending}"
	masked_tokens = []
	orig_tokens = orig_sentence.split()
	new_tokens = []

	idx = orig_tokens.index("necessarily")

	## masking everything after "necessarily"

	new_tokens = orig_tokens[:idx+1] # everything up to necessarily
	tokens_after_nec = orig_tokens[idx+1:]

	nec_id_in_MLM_tokens, MLM_tokens = model.get_token_id(orig_sentence, "necessarily")
	n_tokens_after_nec = len(MLM_tokens) - 1 - nec_id_in_MLM_tokens
	new_tokens += n_tokens_after_nec * [mask_token]

	masked_sentence = ' '.join(new_tokens)

	return masked_sentence

def score_endings(prompt, ending0, ending1, model_name, model):
	if "gpt" in model_name:
		if "gpt3" not in model_name:
			score0 = model.score_conditional(prompt, ending0)
			score1 = model.score_conditional(prompt, ending1)
		else:
			score0 = score_ending_gpt3(prompt, ending0, model_name)
			score1 = score_ending_gpt3(prompt, ending1, model_name)
	else:
		# MLM
		masked0 = mask(prompt, ending0, model_name, model)
		masked1 = mask(prompt, ending1, model_name, model)

		score0 = model.score_masked(orig_sentence=f"{prompt} {ending0}", mask_sentence=masked0)
		score1 = model.score_masked(orig_sentence=f"{prompt} {ending1}", mask_sentence=masked1)

	return score0, score1


def score_prompts(prompt0, prompt1, ending, model_name, model):
	if "gpt" in model_name:
		if "gpt3" not in model_name:
			score0 = model.score_conditional(prompt0, ending)
			score1 = model.score_conditional(prompt1, ending)
		else:

			score0 = score_ending_gpt3(prompt0, ending, model_name)
			score1 = score_ending_gpt3(prompt1, ending, model_name)
	else:
		# MLM
		masked0 = mask(prompt0, ending, model_name, model)
		masked1 = mask(prompt1, ending, model_name, model)
		try:
			score0 = model.score_masked(orig_sentence=f"{prompt0} {ending}", mask_sentence=masked0)
			score1 = model.score_masked(orig_sentence=f"{prompt1} {ending}", mask_sentence=masked1)
		except:
			return None, None

	return score0, score1

def score_two_single_sents(sentence0, sentence1, model_name, model):
	if "gpt" in model_name:
		if "gpt3" not in model_name:
			score0 = model.score_single(sentence0)
			score1 = model.score_single(sentence1)
		else:
			score0 = score_ending_gpt3("", sentence0, model_name)
			score1 = score_ending_gpt3("", sentence1, model_name)
	else:
		score0 = model.score_single(sentence0)
		score1 = model.score_single(sentence1)
	return score0, score1

if __name__ == "__main__":
	model_class = ["gpt2", "mlm", "gpt3"][2]
	if model_class == "gpt2":
		model_names = ["gpt2",
		              "gpt2-medium",
		              "gpt2-large",
		              "gpt2-xl"
		              ][3]
	elif model_class == "mlm":
		model_names = ["bert-base-uncased",
		              "bert-large-uncased",
		              "bert-base-cased",
		              "bert-large-cased",
		              "roberta-base",
		              "roberta-large"
		              ][:4]
	elif model_class == "gpt3":
		model_names = ["gpt3-ada",
		              "gpt3-curie",
		              "gpt3-davinci"][1:]
		openai.api_key = os.getenv("OPENAI_API_KEY")

	#config

	for model_name in model_names:
		print(model_name)
		exceptions = 0

		if model_class == "gpt2":
			model = PPLScorer(model_name)
		elif model_class == "mlm":
			model = MLMScorer(model_name)
		else:
			model = None

		sanity_data_frn = f"data/sanity_check/test.csv"
		rows, fieldnames = read_data(sanity_data_frn)

		new_rows = []

		gold_labels, pred_labels = [], []
		for row in rows:
			prompt0, prompt1, ending, gold_label = row["prompt0"], row["prompt1"], row["ending"], int(row["label"])
			score0, score1 = score_prompts(prompt0, prompt1, ending, model_name, model)

			if score0 == None:
				exceptions += 1
				continue

			if score0 < score1:
				pred_label = 0
			elif score0 > score1:
				pred_label = 1
			else:
				pred_label = gold_label
				print("Equally likely.")

			gold_labels.append(gold_label)
			pred_labels.append(pred_label)

			new_row = row.copy()
			new_row["pred"] = pred_label
			new_row["score0"], new_row["score1"] = score0, score1
			new_rows.append(new_row)

		compute_scores(n_classes=2, gold_labels=gold_labels, pred_labels=pred_labels)

		output_dir = "output_dir/sanity_check"
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)
		fwn = f"{output_dir}/{model_name}_{sanity_data_frn}"
		new_fieldnames = fieldnames + ["pred", "score0", "score1"]
		write_preds(fwn, new_rows, new_fieldnames)
		print(f"exceptions: {exceptions}\n")