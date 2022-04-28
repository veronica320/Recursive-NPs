import os
import numpy as np
import csv
import math

cuda = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from transformers import BertTokenizer, BertLMHeadModel, BertConfig
from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig

root_dir = "/path/to/Recursive-NPs"
os.chdir(root_dir)

cache_dir = '/transformers/cache/dir/'

supported_model_names = ["gpt2",
                         "gpt2-medium",
                         "gpt2-large",
                         "gpt2-xl",
                         ]

class PPLScorer():
	"""A LM scorer for the conditional probability of an ending given a prompt."""
	
	def __init__(self, model_name):

		assert(model_name in supported_model_names)

		self.model_name = model_name
		if "gpt2" in model_name:
			with torch.no_grad():
				self.LM = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir).to("cuda:0")
				self.LM.eval()
			self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)

		elif "roberta" in model_name:
			with torch.no_grad():
				config = RobertaConfig.from_pretrained(model_name)
				config.is_decoder = True
				self.LM = RobertaForCausalLM.from_pretrained(model_name, config=config, cache_dir=cache_dir).to("cuda:0")
				self.LM.eval()
			self.tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

		elif "bert" in model_name:
			with torch.no_grad():
				config = BertConfig.from_pretrained(model_name)
				config.is_decoder = True
				self.LM = BertLMHeadModel.from_pretrained(model_name, config=config, cache_dir=cache_dir).to("cuda:0")
				self.LM.eval()
			self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

		else:
			raise ValueError(f"Unknown model name: {model_name}.")

	def get_input_ids(self, sentence, is_ending = False):
		tokens = self.tokenizer.tokenize(sentence)
		if "gpt" in self.model_name or "roberta" in self.model_name:
			if is_ending:
				tokens[0] = 'Ġ' + tokens[0]
		return tokens

	def score_single(self, sentence):
		tokens = self.get_input_ids(sentence)

		input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)]).to("cuda:0")

		with torch.no_grad():
			outputs = self.LM(input_ids, labels=input_ids)
			log_likelihood = outputs[0].detach().item()

		ppl = math.exp(log_likelihood)
		return log_likelihood


	def score_conditional(self, prompt, ending):
		prompt_tokens = self.get_input_ids(prompt, is_ending=False)
		ending_tokens = self.get_input_ids(ending, is_ending=True)

		all_tokens = prompt_tokens + ending_tokens
		input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(all_tokens)]).to("cuda:0")
		target_ids = input_ids.clone()
		target_ids[:, :len(prompt_tokens)] = -100

		with torch.no_grad():
			outputs = self.LM(input_ids, labels=target_ids)
			log_likelihood = outputs[0].detach().item()

		ppl = math.exp(log_likelihood)

		return log_likelihood