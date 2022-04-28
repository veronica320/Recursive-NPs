#!/usr/bin/env bash

## MPE
python finetune_gpt3.py --data MPE --format concat --model ada
python finetune_gpt3.py --data MPE --format concat --model curie

openai api fine_tunes.results -i ft-eG7mK6qCqLi6jt4rqyDmhIh0 > MPE_ada_results.csv
openai api fine_tunes.results -i ft-h1vkP97guCyCAjtDZG77fQV2 > MPE_curie_results.csv

## ADEPT
python finetune_gpt3.py --data ADEPT --model ada
python finetune_gpt3.py --data ADEPT --model curie

openai api fine_tunes.results -i ft-VnNdy9gMO7DKv8bYuDMkpAor > ADEPT_ada_results.csv
openai api fine_tunes.results -i ft-HBfFr16NuGe9lB8WPiokhiOS > ADEPT_curie_results.csv