import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.system("python run_inference.py --external --max_new_tokens 3")
# os.system("python run_inference.py --cot --wo_unbiased_instruc --max_new_tokens 512")
# os.system("python run_inference.py --wo_unbiased_instruc --max_new_tokens 3")
# os.system("python run_faithful.py --wo_unbiased_instruc --max_new_tokens 3")
# os.system("python run_faithful.py --cot --wo_unbiased_instruc --max_new_tokens 512")

# os.system("python run_inference.py --external --max_new_tokens 3")
# os.system("python run_faithful.py --external --max_new_tokens 3")
# os.system("python run_inference.py --cot --wo_unbiased_instruc --max_new_tokens 512")
os.system("python run_faithful.py --cot --wo_unbiased_instruc --max_new_tokens 512")
