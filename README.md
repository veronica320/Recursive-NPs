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
    - `Qa`: Code for research question (a): Is the knowledge of how to interpret recursive NPs present in LMs?
    - `Qb`: Code for research question (b): Is such knowledge learnable with appropriate
data?
    - `Qc`: Code for research question (c): What can LMs learn from RNPC?
    - `Qd`: Code for research question (d): Is RNPC useful for downstream tasks?

- `output_dir`: Model prediction outputs.

- `output_model_dir`: Finetuned models.

- `env.yml`: The conda environment config file.

## Usage

### (a) Is the knowledge of how to interpret recursive NPs present in LMs?
### (b) Is such knowledge learnable with appropriate
data?
### (c) What can LMs learn from RNPC?
### (d) Is RNPC useful for downstream tasks?

## Citation
If you find this repository useful, please cite our paper:

```
@inproceedings{lyu-etal-2022-my,
      title={Is "my favorite new movie" my favorite movie? Probing the Understanding of Recursive Noun Phrases}, 
      author={Qing Lyu and Hua Zheng and Daoxin Li and Li Zhang and Marianna Apidianaki and Chris Callison-Burch},
	  booktitle = "Proceedings of the 2022 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies (Long and Short Papers)",
      month = jul,
      year = "2022",
      address = "Seattle, Washington, USA",
      publisher = "Association for Computational Linguistics"
}
```