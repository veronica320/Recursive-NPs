import numpy as np

def entailment(model, tokenizer, model_name, premise, hypothesis):
	'''Predicts whether the premise entails (1 yes, 0 no) the hypothesis.'''

	# textattack models have class 1 as entailment
	if "textattack" in model_name:
		entail_idx = 1
	# other models have class 2 as entailment
	else:
		entail_idx = 2

	x = tokenizer.encode(premise, hypothesis, return_tensors='pt')
	logits = model(x.to("cuda:0"))[0]

	probs = logits.softmax(dim=1)[0]
	probs = probs.cpu().detach().numpy()

	entail_prob = probs[entail_idx]
	# summing up the probs for contradiction and neutral, as the prob for non-entailment
	nonentail_prob = sum([prob for i,prob in enumerate(probs) if i != entail_idx])

	if entail_prob >= nonentail_prob:
		return entail_prob, 1  # entail
	else:
		return nonentail_prob, 0  # non-entail

def event_plausibility(model, tokenizer, model_name, first_event, second_event):
	'''Predicts whether second_event is more (2)/ equally (1)/ less likely (0) than first_event.'''

	x = tokenizer.encode(first_event, second_event, return_tensors='pt')
	logits = model(x.to("cuda:0"))[0]

	probs = logits.softmax(dim=1)[0]
	probs = probs.cpu().detach().numpy()

	pred_label = np.argmax(probs)
	confidence = probs[pred_label]

	return confidence, pred_label

