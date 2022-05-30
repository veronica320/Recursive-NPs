'''
A zero-shot harm detection model based on finetuning on an RNPC task (SPTE or EPC).
'''
import os
import sys
from source.Qd.NP_extractor import extract_NPs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class HarmDetector:
	def __init__(self,
	             config
	             ):
		self.RNPC_task = eval(config.RNPC_task)
		self.keywords = self.load_keywords()
		self.cache_dir = config.cache_dir
		print("Loading models...")
		self.load_models()

	def load_keywords(self):
		keywords = []
		frn = "data/harm_detection/keyword_list.txt"
		with open(frn, 'r') as fr:
			for line in fr:
				keywords.append(line.strip())
		return keywords

	def load_models(self):
		model_dict = {"SPTE": "veronica320/SPTE_roberta-large-mnli_all",
		              "EPC": "veronica320/EPC_ADEPT_roberta-l_all"
		              }

		model_name = model_dict[self.RNPC_task]
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=self.cache_dir).to(f"cuda:0")
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)

	def infer_with_model(self, NP, keyword):

		# infer with SPTE model
		if self.RNPC_task == "SPTE":
			premise = f"This is {NP}."
			hypothesis = f"This is a {keyword}."

			x = self.tokenizer.encode(premise, hypothesis, return_tensors='pt')
			logits = self.model(x.to("cuda:0"))[0]
			probs = logits.softmax(dim=1)[0]
			probs = probs.cpu().detach().numpy()

			entail_prob = probs[2]
			nonentail_prob = sum([prob for i, prob in enumerate(probs) if i != 2])

			if entail_prob >= nonentail_prob:
				return "harmful", entail_prob, probs  # entail
			else:
				return "unharmful", nonentail_prob, probs  # non-entail

		# infer with EPC model
		else:
			event1 = f"A {keyword} is harmful."
			event2 = f"{NP} is harmful."

			x = self.tokenizer.encode(event1, event2, return_tensors='pt')
			logits = self.model(x.to("cuda:0"))[0]

			probs = logits.softmax(dim=1)[0]
			probs = probs.cpu().detach().numpy()

			pred_label = np.argmax(probs)
			confidence = probs[pred_label]

			if pred_label in [1, 2]: # equally likely, more likely
				return "harmful", confidence, probs
			else: # less likely
				return "unharmful", confidence, probs

	def predict(self, query):

		cand_NP_keyword_dict = {}

		all_NPs = self.get_NPs(query)
		for NP in all_NPs:
			tokens = NP.split()
			for token in tokens:
				if token.lower() in self.keywords:
					cand_NP_keyword_dict[NP] = token.lower()
		output_dict = {}
		for cand_NP, keyword in cand_NP_keyword_dict.items():
			label, confidence, probs = self.infer_with_model(cand_NP, keyword)
			output_dict[label] = {"confidence": confidence,
			                      "provenance": cand_NP,
			                      "probs": probs
			                      }
		return output_dict

	def get_NPs(self, query):
		return extract_NPs(query)

