import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.system("python run_inference.py --external --wo_unbiased_instruc --max_new_tokens 3")
os.system("python run_faithful.py --external --wo_unbiased_instruc --max_new_tokens 3")
os.system("python run_inference.py --cot --external --wo_unbiased_instruc --max_new_tokens 512") # 到这
os.system("python run_faithful.py --external --max_new_tokens 3")
os.system("python run_faithful.py --cot --external --wo_unbiased_instruc --max_new_tokens 512")
os.system("python run_faithful.py --cot --external --max_new_tokens 512")