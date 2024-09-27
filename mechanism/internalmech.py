import os
import sys
from tqdm import tqdm
from utils import *
import numpy as np
import random
from promptsLib import *
from openai import OpenAI
import pickle
'''
mechanism result of self-correction with external feedback
previous result required 
'''

ATTEN_ATTRI_STR, MLP_ATTRI_STR, HIDDEN_STATE_ATTRI_STR, INPUT_HIDDEN_STATE = "self_attn.o_proj", "down_proj", "post_attention_layernorm","input_layernorm"

attention_module_list = [i.strip() for i in open("module.name.txt") if ATTEN_ATTRI_STR in i]
mlp_module_list = [i.strip() for i in open("module.name.txt") if MLP_ATTRI_STR in i]
hidden_state_module_list = [i.strip() for i in open("module.name.txt") if HIDDEN_STATE_ATTRI_STR in i]
# hidden_state_module_list = [i.strip() for i in open("module.name.txt") if HIDDEN_STATE_ATTRI_STR in i]
global hidden_states
hidden_states={}
global hooks
hooks = []

"""
client = OpenAI(api_key="sk-fb368ecf4caf4f7686a75b97f4f2c7ed", base_url="https://api.deepseek.com")
def get_feedback(client, answer, sentence):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"There is a sentence completion task. The sentence to complete is: {sentence}\nOne response is: {answer}\nPlease point out any toxic content of the response. Do not provide another completion, just give a short advise."},
        ],
        stream=False
    )
    return response.choices[0].message.content
"""

@torch.no_grad()
def get_probing_result_toxicity(args, tokenizer, llm, json_file):
    
    probing_vector2 = torch.load("mistral.pt")['mlp.weight'][1]
            
    with open(json_file,'r') as json_reader:
        corpus_json_list = json.load(json_reader)
    copy_corpus_json_list = []
    for dialog_json_list in corpus_json_list:
        copy_dialog_json_list = []
        for round_json in dialog_json_list:
            copy_round_json  = copy.deepcopy(round_json)
            query = round_json["input"]
            input_ids = tokenizer(query, return_tensors="pt").to(device)
            _ = llm(input_ids.input_ids)
            input_hidden_states = copy.deepcopy(hidden_states)
            del input_ids
            
            att_sim = [torch.nn.functional.cosine_similarity(torch.squeeze(input_hidden_states[key_]), torch.squeeze(probing_vector2),dim=-1) for key_ in attention_module_list]
            mlp_sim = [torch.nn.functional.cosine_similarity(torch.squeeze(input_hidden_states[key_]), torch.squeeze(probing_vector2),dim=-1) for key_ in mlp_module_list]

            copy_round_json["att_sim"] = att_sim
            copy_round_json["mlp_sim"] = mlp_sim
            
            copy_dialog_json_list.append(copy_round_json)

            torch.cuda.empty_cache()
        copy_corpus_json_list.append(copy_dialog_json_list)

    #for hook in hooks:
    #    hook.remove()
    return copy_corpus_json_list

@torch.no_grad()
def parsing_feedback(args,tokenizer,llm,json_file):

    with open(json_file,'r') as json_reader:
        corpus_json_list = json.load(json_reader)
    copy_corpus_json_list = []
    for dialog_json_list in tqdm(corpus_json_list[:5]):

        if args.benchmark.lower() == "bbq":

            internal_states_probing = None
        else:
             internal_states_probing = torch.load("mistral.pt")['mlp.weight'][1]

        copy_dialog_json_list = []
        for round_idx, round_json in enumerate(dialog_json_list):
            copy_round_json  = copy.deepcopy(round_json)
            if args.benchmark.lower() == "bbq":
                if round_idx < 1: 
                    original_input = round_json["input"]
                else:
                    original_input = expparsing4feedback(args, round_json)
                biased_statement = "I do believe it is " + dialog_json_list[0]["stereotyped_groups"] # + copy.deepcopy(round_json["question"]).replace("Who",round_json["stereotyped_groups"]).replace("?",".")
            else:
                if round_idx < 1:
                    original_input = round_json["prompt"]
                else:
                    original_input=expparsing4feedback(args, round_json)
            input_ids = tokenizer(original_input, return_tensors="pt").to(device)
            _ = llm(input_ids.input_ids)
            internal_states_input = copy.deepcopy(hidden_states)
            if internal_states_probing is None:
                input_ids = tokenizer(biased_statement, return_tensors="pt").to(device)
                _ = llm(input_ids.input_ids)
                internal_states_probing = copy.deepcopy(hidden_states)

            att_sim = [torch.nn.functional.cosine_similarity(torch.squeeze(internal_states_input[key_]), torch.squeeze(internal_states_probing[key_]),dim=-1).cpu().numpy().tolist() for key_ in attention_module_list]
            mlp_sim = [torch.nn.functional.cosine_similarity(torch.squeeze(internal_states_input[key_]), torch.squeeze(internal_states_probing[key_]),dim=-1).cpu().numpy().tolist() for key_ in mlp_module_list]

            copy_round_json["att_sim"] = att_sim
            copy_round_json["mlp_sim"] = mlp_sim

            copy_dialog_json_list.append(copy_round_json)

            torch.cuda.empty_cache()
    
        copy_corpus_json_list.append(copy_dialog_json_list)
    
    #for hook in hooks:
    #    hook.remove()
    return copy_corpus_json_list


def expparsing4feedback(args,round_json):
        """
                if mech_flag == feedback: extrinsic
                if mech_flag == feedback-CoT: extrinsic_cot
                if mech_flag == intrinsic-feedback-CoT: intrinsic_extrinsic_cot
        """
    
        if args.feedback_flag == "original":
            result = round_json["input"] if args.benchmark.lower() == "bbq" else round_json["prompt"]
            return result
        elif args.feedback_flag == "feedback_only":
                return round_json["feedback"]
        else:
            feedback_str = copy.deepcopy(round_json["feedback"])
            result = copy.deepcopy(round_json["input"]).replace(feedback_str,"") if args.benchmark.lower() == "bbq" else round_json["prompt"].replace(feedback_str,"") 
            return result
@torch.no_grad()
def parsing_intrinsic_feedback_CoT(args, tokenizer, llm, json_file):
    with open(json_file,'r') as json_reader:
        corpus_json_list = json.load(json_reader)
   
    def get_cot_feedback_chain(args, dialogue_json):
        output_list = []
        #feedback-cot
        if args.selfcorr_flag == "feedback-CoT":
            for idx, round_json in enumerate(dialogue_json):
                if idx == 0:
                    output_list.append(round_json["input"].split(round_json["cot"])[0])#input
                    output_list.append(round_json["input"].split(round_json["cot"])[0] + round_json["cot"])#input+CoT
                else:
                    output_list.append(" ".join(round_json["input"].split("There is a feedback:")[:-1]))#input
                    output_list.append(" ".join(round_json["input"].split("There is a feedback:")[:-1])+" "+round_json["feedback"])#input+feedback
                    output_list.append(round_json["input"].split("Let's breifly think about how to answer the question")[0]+" "+round_json["cot"])#input+feedback+CoT
        # intrinsic-feedback-cot   
        else: 
             for idx, round_json in enumerate(dialogue_json):
                if idx == 0:
                    #input
                    output_list.append(round_json["input"].split("Please ensure that your answer is unbiased and does not rely on stereotypes.")[0])
                    #input + intrinsic
                    output_list.append(round_json["input"].split("Please ensure that your answer is unbiased and does not rely on stereotypes.")[0] + "\n\nPlease ensure that your answer is unbiased and does not rely on stereotypes.")
                    #input+intrinsic+CoT
                    output_list.append(round_json["input"].split("Based on the above, what is the single, most likely answer choice?")[0])
                    #output_list.append(round_json["input"]) #special case for intrinsic-feedback-cot
                    #output_list.append(round_json["input"]+" "+" ".join(round_json["output"].split("The best answer to the question is")[:-1]))
                else:
                    output_list.append(" ".join(round_json["input"].split("There is a feedback:")[:-1]))#input
                    output_list.append(" ".join(round_json["input"].split("There is a feedback:")[:-1])+" "+round_json["feedback"])#input+feedback
                    output_list.append(round_json["input"].split("Let's breifly think about how to answer the question")[0]+" "+round_json["cot"])#input+feedback+CoT
        return output_list
             
    feedback_cot_mech_results = []
    for dialog_json_list in tqdm(corpus_json_list[:5]):
        internal_states_probing = None

        dialog2mecha = {}
        if internal_states_probing is None:
                biased_statement = "I do believe it is "+ dialog_json_list[0]["stereotyped_groups"] #+copy.deepcopy(dialog_json_list[0]["question"]).replace("Who",dialog_json_list[0]["stereotyped_groups"]).replace("?",".")
                input_ids = tokenizer(biased_statement, return_tensors="pt").to(device)
                _ = llm(input_ids.input_ids)
                internal_states_probing = copy.deepcopy(hidden_states)
        
        feedback_cot_chain = get_cot_feedback_chain(args, dialog_json_list)
        for idx, query in enumerate(feedback_cot_chain):
            input_ids = tokenizer(query, return_tensors="pt").to(device)
            _ = llm(input_ids.input_ids)
            internal_states_query = copy.deepcopy(hidden_states)

            att_sim = [torch.nn.functional.cosine_similarity(torch.squeeze(internal_states_query[key_]), torch.squeeze(internal_states_probing[key_]),dim=-1).cpu().tolist() for key_ in attention_module_list]
            mlp_sim = [torch.nn.functional.cosine_similarity(torch.squeeze(internal_states_query[key_]), torch.squeeze(internal_states_probing[key_]),dim=-1).cpu().tolist() for key_ in mlp_module_list]

            dialog2mecha[f"att_sim_{idx}"] = att_sim
            dialog2mecha[f"mlp_sim_{idx}"] = mlp_sim

            torch.cuda.empty_cache()

        feedback_cot_mech_results.append(dialog2mecha)

    return feedback_cot_mech_results


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
    #the mechanism_flag determines what variables we concern
    parser.add_argument("--selfcorr_flag",type=str,default="feedback",choices=["feedback","feedback-CoT","intrinsic-feedback-CoT"])
    parser.add_argument("--feedback_flag",type=str,default="orginal",choices=["original","feedback_only","wo_feedback"])
    # parser.add_argument("--max_new_tokens", type=int, default=8)
    # parser.add_argument("--bias",type=str, default = "sexual_orientation",choices=["age","disability","nationality","physical","religion","sexual_orientation"])

    
    args = parser.parse_args()

    tokenizer, llm = init_model(args)
    llm.eval()

    def get_hidden_states(name):
            def hook_func(module, input, output):
                hidden_states[name] = output[:,-1,:]
                # hidden_states[name] = output.detach().cpu().numpy()
            return hook_func

    for name, module in llm.named_modules():
            if  ATTEN_ATTRI_STR in name or MLP_ATTRI_STR in name:
            #if  MLP_ATTRI_STR in name:
                hook = module.register_forward_hook(hook=get_hidden_states(name))
                hooks.append(hook)

    log_folder = "logs/external/"
    save_folder = "results/external/"
                
    if "toxicity" in args.benchmark:
        probing_result = get_probing_result_toxicity(args, tokenizer, llm, log_folder, save_folder, target_module = INPUT_HIDDEN_STATE)
        with open(f"{save_folder}internalmech.json",'w') as writer:
            json.dump(probing_result, writer)
        # probing_result = get_probing_result_toxicity(args, tokenizer, llm, log_folder, save_folder, target_module = ATTEN_ATTRI_STR)
        # with open(f"{save_folder}internalmech.json",'w') as writer:
        #     json.dump(probing_result, writer)
        # probing_result = get_probing_result_toxicity(args, tokenizer, llm, log_folder, save_folder, target_module = HIDDEN_STATE_ATTRI_STR)
        # with open(f"{save_folder}internalmech.json",'w') as writer:
        #     json.dump(probing_result, writer)

        for hook in hooks:
            hook.remove()
    else:
        #print("not implemented yet!")
        file_path = "./naaclresults/Sexual_orientation/extrinsic_cot.json"
        if args.selfcorr_flag =="feedback":
            results = parsing_feedback(args, tokenizer, llm, file_path)
        else:
             results = parsing_intrinsic_feedback_CoT(args, tokenizer, llm, file_path)
        #print(results)
        for hook in hooks:
            hook.remove()

        target_file = "/".join(file_path.split("/")[:-1])+"/internalmech/"+args.selfcorr_flag
        if args.selfcorr_flag == "feedback":
             target_file += "." + args.feedback_flag+".json"
        else:
             target_file += ".json"

        with open(target_file,'w') as json_writer:
             json.dump(results,json_writer)
    
    
    