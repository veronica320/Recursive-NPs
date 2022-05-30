# Recursive-NPs
Code and data acccompanying the NAACL 2022 paper <Is "my favorite new movie" my favorite movie? Probing the Understanding of Recursive Noun Phrases>.

## Get started

### Environment

Create a conda virtual environment using the provided `env.yml` according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## File Structure

- `data/`:
    - `RNPC`: Our Recursive Noun Phrase Challenge (RNPC) dataset.
        - `modifier_lexicon`: The modifier lexicon we collected. There are three classes of modifiers: intersective, subsective, privative. See the README in the folder for more details.
        - `tasks`: Data for the three RNPC tasks -- Single-Premise Textual Entailment (SPTE), Multi-Premise Textual Entailment (MPTE), Event Plausibility Comparison (EPC). See the README in the folder for more details.
        - `inoculation`: Data for the "inoculation by finetuning" experiment in Section 6 of the paper. See the README in the folder for more details.
    - `existing_benchmarks`: Existing benchmark datasets used for finetuning in Section 5 of the paper, converted to our needed format.
        - `ADEPT`: The ADEPT dataset from [(Emami et al., 2021)](https://github.com/aemami1/ADEPT). 
        - `MPE`: The MPE dataset from [(Lai et al., 2017)](https://github.com/aylai/MultiPremiseEntailment).
        - For MNLI and SNLI, we use existing finetuned models from huggingface, so we don't need the datasets here.
    - `harm_detection`: The harm detection dataset for extrinsic evaluation.  See the README in the folder for more details.
- `source/`: The source code.
    - `Qa`: Code for research question (a) (§5): Is the knowledge of how to interpret recursive NPs present in LMs?
        - `finetune_on_benchmark`: Finetuning Transformer models on existing benchmarks of the same format. 
            - `gpt3`: Finetuning GPT3 on existing benchmarks (since it has its own API). 
            - `other_models`: Finetuning other models (BERT, RoBERTa) on existing benchmarks. 
        - `eval_on_RNPC`: Evaluating the finetuned models on RNPC.
            - `gpt3`: Evaluating GPT3 on RNPC (since it has its own API). 
            - `other_models`: Evaluating other models (BERT, RoBERTa) on RNPC.
    - `Qb`: Code for research question (b) (§6): Is such knowledge learnable with appropriate data?
    - `Qc`: Code for research question (c) (§7): What can LMs learn from RNPC?
    - `Qd`: Code for research question (d) (§8): Is RNPC useful for downstream tasks?

- `output_dir/`: Model prediction outputs.
    - `RNPC/`:
        - `eval_models_ft_on_benchmark/`: Predictions of models finetuned on existing benchmarks and evaluated on RNPC.
        - `inoculation/`: Predictions of models finetuned on RNPC and evaluated on RNPC.
        
- `output_model_dir/`: Finetuned models.

- `env.yml`: The conda environment config file.

## Usage

### Research Question (a): Is the knowledge of how to interpret recursive NPs present in LMs?

In §5 of the paper, we take SOTA models finetuned on existing benchmark(s) of the same format as each RNPC task, and evaluate them on RNPC. 

The following steps allow you to reproduce our experiments.
If you want to finetune the models yourself, start from Step 1. Otherwise, start from Step 2.

#### 1. Finetune a model on an existing benchmark

If you want to finetune **Huggingface models** (BERT, RoBERTa, ...):

(1) Go to the folder
```
cd source/Qa/eval_on_RNPC/finetune_on_benchmark/other_models
```

(2) Set the relevant configuration in `finetune.py`

See comments inside.

(3) Run the finetuning script
```
./finetune.sh
```
The running log will be saved under `logs/` in the current directory. The finetuned model will be saved under `output_model_dir` in the root directory.



Similarly, if you want to finetune **GPT-3 models**:

(1) Go to the folder
```
cd source/Qa/eval_on_RNPC/finetune_on_benchmark/gpt3
```

(2) Set the relevant configuration in `finetune_gpt3.py`

See comments inside.

(3) Set the environment variable `OPENAI_API_KEY`
```
export OPENAI_API_KEY = [YOUR KEY FROM https://beta.openai.com/account/api-keys]
```

(4) Run the finetuning script
```
./finetune_gpt3.sh
```
The running log will be saved under `logs/` in the current directory. 
The finetuned model be saved on the OpenAI server, which can be accessed using the model name. The model name will be printed to the log file.


#### 2. Evaluate the finetuned model on an RNPC task


If you want to evaluate **Huggingface models** (BERT, RoBERTa, ...):

(1) Go to the folder
```
cd source/Qa/eval_on_RNPC/other_models
```

(2) Set the relevant configuration in `config.json`
- `cache_dir`: the directory to store model cache
- `gpu_devices`: the index of cuda device to use
- `task`: which task to evaluate on

(3) Run the evaluation script
```
./eval_ft_models_on_RNPC.sh
```
The performance of all available models on the specified task (see `eval_ft_models_on_RNPC.py` line 82 for a list of available models) will be printed, e.g.
```
Evaluating models on MPTE: ...
MPE_bert
{'accuracy': 0.472, 'prediction': 0.48, 'recall': 0.44, 'f1': 0.459, 'weighted-F1': 0.472}
MPE_bert-l
{'accuracy': 0.415, 'prediction': 0.342, 'recall': 0.163, 'f1': 0.221, 'weighted-F1': 0.373}
MPE_roberta
{'accuracy': 0.511, 'prediction': 0.51, 'recall': 1.0, 'f1': 0.675, 'weighted-F1': 0.347}
MPE_roberta-l
{'accuracy': 0.509, 'prediction': 0.509, 'recall': 1.0, 'f1': 0.675, 'weighted-F1': 0.343}
```
The model predictions will be saved to `output_dir/RNPC/eval_models_ft_on_benchmark`, under the specified task folder.


Similarly, if you want to evaluate **GPT-3 models**:

(1) Go to the folder
```
cd source/Qa/eval_on_RNPC/gpt3
```
(2) Set the relevant configuration in `config.json`. Most fields are the same as the previous case, but there are a few more:
- `gpt3_model`：which version of gpt-3 to evaluate (ada or curie). We don't include davinci since it wasn't been open for finetuning at the time of the paper.
- `trial`: whether this is a trial run. If true, then the evaluation will run on only 5 examples. This is for debugging purposes since running GPT-3 is costly.  

(3) Set the environment variable `OPENAI_API_KEY`
```
export OPENAI_API_KEY = [YOUR KEY FROM https://beta.openai.com/account/api-keys]
```
(4) Run the evaluation script
```
./eval_ft_gpt3_on_RNPC.sh
```
The output will look similar to the previous case.


### Research Question (b): Is such knowledge learnable with appropriate data?

In §6 of the paper, we finetune models from (a) on a small number of examples (10-200) from each RNPC task and and evaluate them on the rest. 

The following steps allow you to reproduce our experiments.
If you want to finetune the models yourself, start from Step 1. Otherwise, start from Step 2.

#### 1. Finetune a model on an RNPC task

(1) Go to the folder
```
cd source/Qb
```

(2) Set the relevant configuration in `finetune_on_RNPC.py`

See comments inside.

(3) Run the finetuning script
```
./finetune_on_RNPC.sh
```
The running log will be saved into `nohup.out` in the current directory. The finetuned model will be saved under `output_model_dir/inoculation/` in the root directory.


#### 2. Evaluate the finetuned model on an RNPC task

We have released the model finetuned on the full 200 training examples for each task. 
If you'd like to try models with other numbers of training examples, please go to Step 1 and finetune them first.

(1) Go to the folder
```
cd source/Qb
```

(2) Set the relevant configuration in `config.json`
- `cache_dir`: the directory to store model cache
- `gpu_devices`: the index of cuda device to use
- `task`: which task to evaluate on
- `n_train_examples`: which model to use (identified by the number of training examples)

(3) Run the evaluation script
```
./eval_ft_models_on_RNPC.sh
```
The performance of the optimal model on the specified task (see the `model_dict` variable in `eval.py` for model names) will be printed, e.g.
```
Evaluating roberta-large-mnli on SPTE...
# Training examples: 200
{'accuracy': 0.917, 'precision': 0.93, 'recall': 0.894, 'f1': 0.912, 'weighted-F1': 0.917}

```
The model predictions will be saved to `output_dir/RNPC/inoculation`, under the specified task folder.



### Research Question (c): What can LMs learn from RNPC?
We use [the code from the Amnesic Probing paper](https://github.com/yanaiela/amnesic_probing) (Elazor et al. 2021). Available upon request.

### Research Question (d): Is RNPC useful for downstream tasks?

In §8 of the paper, we demonstrate the knowledge learned from RNPC is useful for extrinsic tasks, e.g. harm detection.
On a small harm detection test set, we evaluate models finetuned on RNPC and compare their performance to GPT-3 variants as a baseline.
All evaluations are done in a zero-shot setting.

The following steps allow you to reproduce our experiments.

#### 1. Evaluate RNPC-based models on harm detection
(1) Go to the folder
```
cd source/Qd/
```
(2) Run the following code
```
python eval_RNPC_models.py
```
The performance of the model will be printed to stdout, e.g., for the EPC-based model:
```
{'accuracy': 0.729, 'precision': 0.664, 'recall': 0.929, 'f1': 0.775, 'weighted-F1': 0.718}
```
The model predictions will be saved under `output_dir/harm_detection/`.

#### 2. Evaluate GPT-3 baselines on harm detection

(1) Go to the folder
```
cd source/Qd/
```
(2) Set the environment variable `OPENAI_API_KEY`
```
export OPENAI_API_KEY = [YOUR KEY FROM https://beta.openai.com/account/api-keys]
```
(3) Run the following code
```
python eval_gpt3.py
```
The output will look similar to the previous case.

## Citation
If you find this repository useful, please cite our paper:

```
@inproceedings{lyu-etal-2022-my,
      title={Is "My Favorite New Movie" My Favorite Movie? Probing the Understanding of Recursive Noun Phrases}, 
      author={Qing Lyu and Hua Zheng and Daoxin Li and Li Zhang and Marianna Apidianaki and Chris Callison-Burch},
	  booktitle = "Proceedings of the 2022 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies (Long and Short Papers)",
      month = jul,
      year = "2022",
      address = "Seattle, Washington, USA",
      publisher = "Association for Computational Linguistics"
}
```