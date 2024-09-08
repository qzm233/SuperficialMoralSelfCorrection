import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.system("python run_inference.py --cot --external --max_new_tokens 300 > logs/bbq_external_cot.log 2>&1")
os.system("python run_faithful.py --cot --external --max_new_tokens 300 > logs/faithful_bbq_external_cot.log 2>&1")
os.system("python run_faithful.py --external --max_new_tokens 3 > logs/faithful_bbq_external.log 2>&1")