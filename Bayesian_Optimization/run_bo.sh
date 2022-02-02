#!/bin/bash

export PYTHONPATH="$(pwd)"

python bo.py \
  --data-name final_structures6 \
  --model 'SVGE' \
  --save-appendix '../state_dicts/SVGE_ENAS/' \
  --checkpoint 300 \
  --res-dir="res/" \
  --device 'cuda:0' \
  --BO_rounds 10 \
  --BO_batch_size 50 