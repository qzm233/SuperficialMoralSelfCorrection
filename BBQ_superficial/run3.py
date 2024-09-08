import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.system("python run_faithful.py --external --wo_unbiased_instruc --max_new_tokens 3 > logs/faithful_bbq_external_wo_unbiased_instruc.log 2>&1")
os.system("python run_inference.py --cot --external --wo_unbiased_instruc --max_new_tokens 300 > logs/bbq_external_cot_wo_unbiased_instruc.log 2>&1")
os.system("python run_faithful.py --cot --external --wo_unbiased_instruc --max_new_tokens 300 > logs/faithful_bbq_external_cot_wo_unbiased_instruc.log 2>&1")