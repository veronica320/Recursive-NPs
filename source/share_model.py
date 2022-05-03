import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
	root_dir = "/shared/lyuqing/Recursive-NPs"
	os.chdir(root_dir)

	API_token = "CHANGE TO YOUR OWN"

	output_model_dir = "output_model_dir"
	model_names = ["MPE_bert-l","MPE_roberta","MPE_roberta-l","ADEPT_bert","ADEPT_bert-l","ADEPT_roberta","ADEPT_roberta-l"][2:]
	for model_name in model_names:
		print(model_name)
		model_path = f"{output_model_dir}/{model_name}"
		model = AutoModelForSequenceClassification.from_pretrained(model_path)
		model.push_to_hub(model_name, use_auth_token = API_token)
		tokenizer = AutoTokenizer.from_pretrained(model_path)
		tokenizer.push_to_hub(model_name)

