import os

os.system("python run_inference.py --max_new_tokens 3") 
os.system("python run_faithful.py --max_new_tokens 3") 
os.system("python run_inference.py --cot --external --max_new_tokens 512") 
os.system("python run_faithful.py --cot --external --max_new_tokens 512")