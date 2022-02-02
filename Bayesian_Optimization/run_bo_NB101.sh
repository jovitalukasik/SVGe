#!/bin/bash

export PYTHONPATH="$(pwd)"

python bo_nb101.py \
    --model_name 'SVGE' \
    --saved_log_dir '../state_dicts/SVGE_NB101/' \
    --share 1 \
    --keep 1000\
    --device 'cuda:0' \
    --BO_rounds 10 \
    --BO_batch_size 50 \

