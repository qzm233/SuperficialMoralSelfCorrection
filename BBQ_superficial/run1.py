import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system("python run_inference.py --cot --max_new_tokens 300")# > logs/bbq_intrinsic_cot.log 2>&1
os.system("python run_faithful.py --cot --max_new_tokens 300 ")#> logs/faithful_bbq_intrinsic_cot.log 2>&1
os.system("python run_inference.py --external --max_new_tokens 3")# > logs/bbq_external.log 2>&1
