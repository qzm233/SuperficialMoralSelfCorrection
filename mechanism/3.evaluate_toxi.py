import sys
sys.path.append('.')
from utils import *
from tqdm import tqdm
import json
import time
from googleapiclient.errors import HttpError
'''
evaluate the toxicity of output using respective api
'''

def evaluate_toxicity_score(rep_file, max_requests_per_minute=60):
    with open(rep_file, "r") as reader:
        rep_list = json.load(reader)

    print(len(rep_list))
    assert len(rep_list) == 2000
    delay = 60 / max_requests_per_minute
    count = 0
    for sample in tqdm(range(len(rep_list)), desc="Processing samples"):
        count += 1
        for round in tqdm(range(len(rep_list[sample])), desc="Processing rounds", leave=False):
            success = False
            if "toxicity_socre" in rep_list[sample][round].keys(): 
                rep_list[sample][round]["toxicity_score"] = rep_list[sample][round]["toxicity_socre"]
                del rep_list[sample][round]["toxicity_socre"]
                continue
            if "toxicity_score" in rep_list[sample][round].keys():
                continue
            while not success:
                try:
                    assert "toxicity_score" not in rep_list[sample][round].keys()
                    print(rep_list[sample][round]["output"])
                    toxic_score = eva_toxicity(rep_list[sample][round]["output"])
                    rep_list[sample][round]["toxicity_score"] = toxic_score
                    success = True
                except HttpError as e:
                    if e.resp.status == 429: 
                        print("Rate limit exceeded. Waiting before retrying...")
                        time.sleep(delay) 
                    else:
                        print("Error:", e)
                        time.sleep(delay)
                        success = True
                time.sleep(delay)  
        if count % 100 == 0:
            with open(rep_file, 'w') as writer:
                json.dump(rep_list, writer)
                print(f"write sample {sample}")
    with open(rep_file, 'w') as writer:
                json.dump(rep_list, writer)
                print(f"write sample {sample}")

# evaluate_toxicity_score("results/intrinsic/internalmech.json")
evaluate_toxicity_score("results/external/internalmech.json")
