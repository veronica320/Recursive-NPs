# Recursive-NPs
Code and data acccompanying the paper in submission <Is "my favorite new movie" my favorite movie? Probing the Understanding of Recursive Noun Phrases>.

## Get started

### Environment

Create a conda virtual environment using the provided `env.yml` according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

### Data

Sample data can be found in the Supplementar Materials and put under `/data`. Full data will be released upon paper publication.

## File Structure

- `data/`: Sample data.
    	
- `source/`: The source code.
    - `Qa`: Code for research question (a): Is the knowledge of how to interpret recursive NPs present in LMs?
    - `Qb`: Code for research question (b): Is the knowledge of how to interpret recursive NPs present in LMs?

- `output_dir`: Model prediction outputs.

- `output_model_dir`: Finetuned models.

- `env.yml`: The conda environment config file.
