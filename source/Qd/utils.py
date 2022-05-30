import openai
import numpy as np

def query(model, prompt, n_classes=5):
	'''Query GPT3 model for prompt-based generation.'''
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
	response_label, top_logprobs = response["choices"][0]["text"].lower().strip(), \
	                               response["choices"][0]["logprobs"]["top_logprobs"][0]
	label_probs = {key.lower().strip():np.exp(value) for key, value in top_logprobs.items()}
	return response_label, label_probs

def gold_label2idx(label):
	assert label in ["GOOD", "HARM"]
	if label == "GOOD": # harmless
		return 0
	else: # harmful
		return 1

def pred_label2idx(label):
	if label in ["yes", "no"]:
		if label == "yes": # harmful
			return 1
		else: # harmless
			return 0
	elif label in ["harmful", "unharmful"]:
		if label == "harmful": # harmful
			return 1
		else: # harmless
			return 0
	else:
		raise ValueError(f"Unrecognized label: {label}")


