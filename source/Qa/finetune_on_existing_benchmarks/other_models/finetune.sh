#!/usr/bin/env bash

## Finetune baselines on MPE
nohup python finetune_te.py --mode train_eval --task MNLI --target MPE --model bert --cuda 0 > logs/mpe_bert.log 2>&1 &
nohup python finetune_te.py --mode train_eval --task MNLI --target MPE --model bert-l --cuda 1 > logs/mpe_bert-l.log 2>&1 &
nohup python finetune_te.py --mode train_eval --task MNLI --target MPE --model roberta --cuda 2 > logs/mpe_roberta.log 2>&1 &
nohup python finetune_te.py --mode test --task MNLI --target MPE --model roberta-l --cuda 6 > logs/mpe_robertal_test.log 2>&1 &

## Finetune baselines on ADEPT
nohup python finetune_te.py --mode train_eval --task WNLI --target ADEPT --model bert --cuda 0 > logs/adept_bert.log 2>&1 &
nohup python finetune_te.py --mode train_eval --task WNLI --target ADEPT --model bert-l --cuda 1 > logs/adept_bert-l.log 2>&1 &
nohup python finetune_te.py --mode train_eval --task WNLI --target ADEPT --model roberta --cuda 2 > logs/adept_roberta.log 2>&1 &
nohup python finetune_te.py --mode train_eval --task WNLI --target ADEPT --model roberta-l --cuda 4 > logs/adept_roberta-l.log 2>&1 &

