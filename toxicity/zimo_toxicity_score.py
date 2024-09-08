import sys
sys.path.append('.')
from utils import *
from tqdm import tqdm
import json
import time
from googleapiclient.errors import HttpError

def zimo_internal_toxicity_score(rep_file, max_requests_per_minute=60):
    files = ["0-500","500-1000","1000-1500","1500-2000"]
    rep_list = []
    for file in files:
        with open(rep_file+file+"/external_realtoxicity.internalmech.two.json","r") as reader:
            rep_list = rep_list + json.load(reader)
    # with open("experiments/intrinsic/total50_intrinsic_realtoxicity.internalmech.json","r") as reader:
    #     rep_list = json.load(reader)
    # with open(rep_file, "r") as reader:
    #     rep_list = json.load(reader)
    print(len(rep_list))
    assert len(rep_list) == 1995
    delay = 60 / max_requests_per_minute
    count = 0
    for sample in tqdm(range(len(rep_list)), desc="Processing samples"):
        count += 1
        # if count <= 50:
        #     continue
        for round in tqdm(range(len(rep_list[sample])), desc="Processing rounds", leave=False):
            success = False
            while not success:
                try:
                    if "toxicity_score" not in rep_list[sample][round].keys():
                        print(rep_list[sample][round]["output"])
                        toxic_score = eva_toxicity(rep_list[sample][round]["output"])
                        rep_list[sample][round]["toxicity_socre"] = toxic_score
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
        if count % 50 == 0:
            with open(rep_file+f"total{count}_external_realtoxicity.internalmech.json", 'w') as writer:
                json.dump(rep_list, writer)
                print(f"write sample {sample}")

    with open(rep_file+f"final{len(rep_list)}_external_realtoxicity.internalmech.json", 'w') as writer:
                json.dump(rep_list, writer)
                print(f"write sample {sample}")

zimo_internal_toxicity_score("experiments/external/")