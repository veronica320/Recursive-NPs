import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
	root_dir = "/shared/lyuqing/Recursive-NPs"
	os.chdir(root_dir)

	API_token = "GET YOUR TOKEN from https://huggingface.co/settings/tokens"

	output_model_dir = "output_model_dir"
	model_names = ["EPC_ADEPT_roberta-l_all","SPTE_roberta-large-mnli_all"]
	for model_name in model_names:
		print(model_name)
		model_path = f"{output_model_dir}/{model_name}"
		model = AutoModelForSequenceClassification.from_pretrained(model_path)
		model.push_to_hub(model_name, use_auth_token = API_token)
		tokenizer = AutoTokenizer.from_pretrained(model_path)
		tokenizer.push_to_hub(model_name)

