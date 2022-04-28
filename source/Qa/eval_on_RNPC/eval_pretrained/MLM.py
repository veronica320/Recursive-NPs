import os
import numpy as np
import csv
import math
import copy
cuda = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
import torch

from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

root_dir = "/path/to/Recursive-NPs"
os.chdir(root_dir)

cache_dir = '/transformers/cache/dir/'

supported_model_names = ["bert-base-uncased",
                         "bert-large-uncased",
                         "bert-base-cased",
                         "bert-large-cased",
                         "roberta-base",
                         "roberta-large"
                         ]


class MLMScorer():
	"""A LM scorer for the conditional probability of an ending given a prompt."""

	def __init__(self, model_name):

		assert (model_name in supported_model_names)

		self.model_name = model_name

		if "roberta" in model_name:
			self.model_type = "roberta"
			with torch.no_grad():
				self.LM = RobertaForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir).to("cuda:0")
				self.LM.eval()
			self.tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

		elif "bert" in model_name:
			self.model_type = "bert"
			with torch.no_grad():
				self.LM = BertForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir).to("cuda:0")
				self.LM.eval()
			self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

		else:
			raise ValueError(f"Unknown model name: {model_name}.")

	def get_token_length(self, text):
		return len(self.tokenizer(text)["input_ids"]) - 2

	def score_single(self, sentence):
		orig_inputs = self.tokenizer(sentence, return_tensors="pt").to("cuda:0")
		labels = orig_inputs.data["input_ids"]
		n_tokens = len(labels)

		total_loss = 0
		for i in range(n_tokens):
			inputs = copy.deepcopy(orig_inputs)
			inputs["input_ids"][:, i] = self.tokenizer.mask_token_id

			with torch.no_grad():
				outputs = self.LM(**inputs, labels=labels)
				loss = outputs.loss.detach().item()
			total_loss += loss

		return total_loss/len(labels)

	def get_token_id(self, text, token):
		tokens = self.tokenizer.tokenize(text)
		target = None
		if self.model_type == "roberta":
			target = "Ġ" + token
		elif self.model_type == "bert":
			target = token
		return tokens.index(target), tokens

	def score_masked(self, orig_sentence, mask_sentence):
		inputs = self.tokenizer(mask_sentence, return_tensors="pt").to("cuda:0")
		labels = self.tokenizer(orig_sentence, return_tensors="pt")["input_ids"].to("cuda:0")

		mask_token_pos = [i for i, token_id in enumerate(inputs.data["input_ids"][0]) if token_id == self.tokenizer.mask_token_id]
		mask_start = mask_token_pos[0]
		mask_end = mask_token_pos[-1]

		for pos in range(len(labels[0])):
			if pos not in mask_token_pos:
				labels[:, pos] = -100

		outputs = self.LM(**inputs, labels=labels)
		loss = outputs.loss.detach().item()
		return loss