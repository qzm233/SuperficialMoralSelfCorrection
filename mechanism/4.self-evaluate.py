import json
from utils import *
from promptsLib import *
from tqdm import tqdm
import random

'''
MCMC QA compare for the outputs
'''

def self_evaluate(args, llm, tokenizer, log_folder, save_folder):
    print("log_folder: ", log_folder)
    print("save_folder: ", save_folder)
    data = load_benchmark(args)
    with open(f"{save_folder}internalmech.json", "r") as reader:
        sample_list = json.load(reader)
    assert len(sample_list) == 2000
    count = 0
    chains_list = []
    for q_idx, sample in tqdm(enumerate(sample_list)):
        output_states = {}
        assert data[q_idx] in sample[0]['input']
        for idx, round_ in enumerate(sample):
            output_states[idx] = round_['output']
        distribution ,raw_chain = evaluate_preference(data[q_idx], output_states, num_loop = 1000)
        print(distribution)
        chains_list.append({
            "states": output_states,
            "chain": raw_chain,
            "distribution": distribution
        }
        )
        print()
        if (q_idx+1) % 50 == 0:
            print("want to save!!!")
            with open(f"{save_folder}chains.json", "w") as writer:
                json.dump(chains_list, writer)
    return chains_list

def evaluate_preference(prompt, output_states, num_loop = 1000):
    # print(output_states)
    distribution = {}
    for i in range(8):
        distribution[i] = 0
    raw_chain = []
    random.seed(24)

    index = random.randint(0, 7)
    raw_chain.append(index)
    distribution[index] += 1
    for round_ in range(num_loop):
        index_tmp = random.randint(0, 7)
        while index_tmp == index:
            index_tmp = random.randint(0, 7)
        index_order = [index, index_tmp]
        random.shuffle(index_order)
        # print(index_order)
        query = self_evaluation_prompt
        if not args.fewshot:
            query = copy.deepcopy(query).replace("#FEWSHOT", "")
        query = copy.deepcopy(query).replace("#QUESTION", prompt)
        query = copy.deepcopy(query).replace("#OUTPUT1", output_states[index_order[0]])
        query = copy.deepcopy(query).replace("#OUTPUT2", output_states[index_order[1]])
        response = get_response(args, tokenizer, llm, query)
        # print(response)
        if response == "(A)":
            raw_chain.append((str(index_order),index_order[0]))
            distribution[index_order[0]] += 1
            index = index_order[0]
        elif response == "(B)":
            raw_chain.append((str(index_order),index_order[1]))
            distribution[index_order[1]] += 1
            index = index_order[1]
        else:
            print(f"failed to compare {index_order}")
            raw_chain.append(f"failed to compare {index_order}")
            index = random.randint(0, 7)
            continue
        # print(f"{index} win between {index_order}")
    return distribution, raw_chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        default="realtoxicity",
        choices=["winogender", "bbq", "realtoxicity", "bbq_debug"],
    )
    parser.add_argument("--llm", type=str, default="mistral")
    parser.add_argument("--cluster", type=str, default="zhiyu")
    parser.add_argument("--correction_type", type = str, default="intrinsic", choices=['intrinsic', 'external'])
    parser.add_argument("--fewshot", action="store_true" )
    parser.add_argument("--max_new_tokens", type=int, default=4)
    # parser.add_argument("--bias",type=str, default = "sexual_orientation",choices=["age","disability","nationality","physical","religion","sexual_orientation"])
    
    args = parser.parse_args()

    tokenizer, llm = init_model(args)
    llm.eval()
    log_folder = f"logs/{args.correction_type}/"
    save_folder = f"results/{args.correction_type}/"
                
    if "toxicity" in args.benchmark:
        result = self_evaluate(args,llm,tokenizer,log_folder,save_folder)
        with open(f"{save_folder}chains.json",'w') as writer:
            json.dump(result, writer)
        # probing_result = get_probing_result_toxicity(args, tokenizer, llm, log_folder, save_folder, target_module = ATTEN_ATTRI_STR)
        # with open(f"{save_folder}internalmech.json",'w') as writer:
        #     json.dump(probing_result, writer)
        # probing_result = get_probing_result_toxicity(args, tokenizer, llm, log_folder, save_folder, target_module = HIDDEN_STATE_ATTRI_STR)
        # with open(f"{save_folder}internalmech.json",'w') as writer:
        #     json.dump(probing_result, writer)
    else:
        print("not implemented yet!")
        
    
    
    
    
