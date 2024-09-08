#!/bin/bash

nohup python run_faithful.py --external --max_new_tokens 3 > logs/faithful_bbq_external.log 2>&1 
nohup python run_faithful.py --external --wo_unbiased_instruc --max_new_tokens 3 > logs/faithful_bbq_external_wo_unbiased_instruc.log 2>&1 
nohup python run_faithful.py --cot --max_new_tokens 300 > logs/faithful_bbq_intrinsic_cot.log 2>&1 