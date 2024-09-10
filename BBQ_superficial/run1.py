import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system("python run_inference.py --max_new_tokens 3")
os.system("python run_inference.py --cot --external --max_new_tokens 512")
os.system("python run_inference.py --cot --max_new_tokens 512")
os.system("python run_faithful.py --max_new_tokens 3")
os.system("python run_faithful.py --cot --max_new_tokens 512")






