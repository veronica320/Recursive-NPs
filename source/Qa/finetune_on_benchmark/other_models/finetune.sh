#!/usr/bin/env bash

## Finetune baselines on MPE
nohup python finetune.py --mode train_eval --task WNLI --target MPE --model bert --cuda 4 > logs/mpe_bert.log 2>&1 &
#nohup python finetune.py --mode train_eval --task WNLI --target MPE --model bert-l --cuda 5 > logs/mpe_bert-l.log 2>&1 &
#nohup python finetune.py --mode train_eval --task WNLI --target MPE --model roberta --cuda 6 > logs/mpe_roberta.log 2>&1 &
#nohup python finetune.py --mode test --task WNLI --target MPE --model roberta-l --cuda 7 > logs/mpe_robertal_test.log 2>&1 &
#
### Finetune baselines on ADEPT
#nohup python finetune.py --mode train_eval --task WNLI --target ADEPT --model bert --cuda 4 > logs/adept_bert.log 2>&1 &
#nohup python finetune.py --mode train_eval --task WNLI --target ADEPT --model bert-l --cuda 5 > logs/adept_bert-l.log 2>&1 &
#nohup python finetune.py --mode train_eval --task WNLI --target ADEPT --model roberta --cuda 6 > logs/adept_roberta.log 2>&1 &
#nohup python finetune.py --mode train_eval --task WNLI --target ADEPT --model roberta-l --cuda 7 > logs/adept_roberta-l.log 2>&1 &

