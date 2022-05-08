#!/usr/bin/env bash

## MPE
python finetune_gpt3.py --data MPE --model ada
python finetune_gpt3.py --data MPE --model curie

## ADEPT
python finetune_gpt3.py --data ADEPT --model ada
python finetune_gpt3.py --data ADEPT --model curie