#!/usr/bin/env bash

## Finetune models on disjoint split
# SPTE
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 10 --model roberta-large-mnli --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 25 --model roberta-large-mnli --cuda 0
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 50 --model roberta-large-mnli --cuda 0
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 75 --model roberta-large-mnli --cuda 0
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 100 --model roberta-large-mnli --cuda 0
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 125 --model roberta-large-mnli --cuda 0
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 150 --model roberta-large-mnli --cuda 0
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 175 --model roberta-large-mnli --cuda 0
#nohup python finetune_te.py --mode train_eval --task WNLI --target SPTE --batch 4 --strategy disjoint_adjs --n_train 200 --model roberta-large-mnli --cuda 0

## MPTE
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 10 --model MPE_roberta --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 25 --model MPE_roberta --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 50 --model MPE_roberta --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 75 --model MPE_roberta --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 100 --model MPE_roberta --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 125 --model MPE_roberta --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 150 --model MPE_roberta --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 175 --model MPE_roberta --cuda 1
#nohup python finetune_te.py --mode train_eval --task WNLI --target MPTE --batch 4 --strategy disjoint_adjs --n_train 200 --model MPE_roberta --cuda 1

## EPC
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4 --strategy disjoint_adjs --n_train 10 --model ADEPT_roberta-l --cuda 2
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4 --strategy disjoint_adjs --n_train 25 --model ADEPT_roberta-l --cuda 2
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4 --strategy disjoint_adjs --n_train 50 --model ADEPT_roberta-l --cuda 2
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4  --strategy disjoint_adjs --n_train 75 --model ADEPT_roberta-l --cuda 2
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4 --strategy disjoint_adjs --n_train 100 --model ADEPT_roberta-l --cuda 2
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4 --strategy disjoint_adjs --n_train 125 --model ADEPT_roberta-l --cuda 2
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4 --strategy disjoint_adjs --n_train 150 --model ADEPT_roberta-l --cuda 2
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4 --strategy disjoint_adjs --n_train 175 --model ADEPT_roberta-l --cuda 2
#python finetune_te.py --mode train_eval --task WNLI --target EPC --batch 4 --strategy disjoint_adjs --n_train 200 --model ADEPT_roberta-l --cuda 2