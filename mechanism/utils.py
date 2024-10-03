from __future__ import division
import os

# os.system("pip install ai2-olmo")
import argparse
import copy
import glob
import json
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
)
import random
from tqdm import tqdm
from googleapiclient import discovery

PERSPECTIVE_API_KEY = "AIzaSyDerXgs_qA3m-e1JUvmoqyr3VlLH-q6Yag"

llama3_token = "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"
import huggingface_hub

# huggingface_hub.login(token=llama3_token)

llama_7b_dir = "/scratch0/liuguan5/llama/llama-2-7b-chat-hf/"

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# device = "cpu"

WINOGENDER_INSTRUCT_STR = (
    "Please ensure that your answer is unbiased and does not rely on stereotypes."
)
BBQ_INSTRUCT_STR = (
    "Please ensure that your answer is unbiased and does not rely on stereotypes."
)
STEREOTYPED_GROUPS = "stereotyped_groups"

def init_model(args):
    cache_dir = ""
    cache_dir_slim = "/home/zhangxit/files/llms"
    cache_dir_psu = "/data/bochuan/DPO/cache"
    cache_dir_voodoo = "/scratch0/liuguan5/pretrained_models"
    cache_dir_grandriver = "/localscratch/haitaomao"
    cache_dir_zhiyu = "/home/zhiyu2/guangliang/zimo/models"
    cache_dir_lei = "../../models"
    if args.cluster == "psu":
        cache_dir = cache_dir_psu
    elif args.cluster == "slim":
        cache_dir = cache_dir_slim
    elif args.cluster == "voodoo":
        cache_dir = cache_dir_voodoo
    elif args.cluster == "grandriver":
        cache_dir = cache_dir_grandriver
    elif args.cluster == "zhiyu":
        cache_dir = cache_dir_zhiyu
    elif args.cluster == "lei":
        cache_dir = cache_dir_lei

    if args.llm == "mistral":  # 7B
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", cache_dir=cache_dir
        )
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", cache_dir=cache_dir
        )
        return tokenizer, model.to(device)

    elif args.llm == "gemma2-9b":
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-9b-it", cache_dir=cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-9b-it", cache_dir=cache_dir, device_map="auto"
        )
        return tokenizer, model

    elif args.llm == "llama3-8B":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=cache_dir
        )

        return tokenizer, model.to(device)

    elif args.llm == "llama3-70B":
        # meta-llama/Meta-Llama-3-70B-Instruct
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-70B-Instruct", cache_dir=cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-70B-Instruct", cache_dir=cache_dir
        )

        return tokenizer, model.to(device)

    elif "gpt2" in args.llm:  # args.llm == "gpt2-xl":
        if args.llm not in ["gpt2", "gpt2-medium", "gpt2-large"]:
            raise RuntimeError
        config = AutoConfig.from_pretrained(
            args.llm, output_hidden_states=True, output_attentions=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.llm, config=config, cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(args.llm, cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model.to(device)

    elif "llama2" in args.llm:  # 70B,13B,7B
        tokenizer = LlamaTokenizer.from_pretrained(args.llama2_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        if args.llm.split("-")[-1].lower() not in args.llama2_path.lower():
            print("*" * 20)
            print("INPUT PATH OF LLAMA2 IS INCORRECT!")
            print("*" * 20)
            raise RuntimeError
        model = LlamaForCausalLM.from_pretrained(
            args.llama2_path,
            device_map="auto",
            output_attentions=True,
            return_dict=True,
        )

        return tokenizer, model

    elif args.llm == "tinyLlama":  # 1B
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir=cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir=cache_dir
        )
        return tokenizer, model.to(device)

    elif args.llm == "phi":  # 3B
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        return tokenizer, model.to(device)

    elif args.llm == "olmo":
        olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B")
        tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-1B")
        return tokenizer, olmo.to(device)

    elif args.llm == "gemma9b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", cache_dir=cache_dir)
        return tokenizer, model.to(device)

def init_nli_model(args):
    cache_dir = ""
    cache_dir_slim = "/home/zhangxit/files/llms"
    cache_dir_psu = "/data/bochuan/DPO/cache"
    cache_dir_zhiyu = "/home/zhiyu2/guangliang/zimo/models"
    if args.cluster == "psu":
        cache_dir = cache_dir_psu
    elif args.cluster == "slim":
        cache_dir = cache_dir_slim
    elif args.cluster == "zhiyu":
        cache_dir = cache_dir_zhiyu

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli", cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli", cache_dir=cache_dir).to(
        device)
    return tokenizer, model

def remove_last_sentence(text):
    sentences = text.split(". ")
    last_idx = len(sentences) - 1
    min_length = min(len(sentences[last_idx]), len(sentences[last_idx - 1]))
    if sentences[last_idx][:min_length] == sentences[last_idx - 1][:min_length]:
        sentences.remove(sentences[last_idx])

    text = ". ".join(sentences)
    return text


def load_winogender():
    dataset = []
    for line in open("../data/winogender.multianswer.txt"):
        question, label = line.strip().split("\t")
        dataset.append({
            "question": question,
            "label":label
        })
    return dataset


def load_bbq(args):
    dataset = []
    print(f"load bbq {args.bias}")
    file = f"data/bbq.{args.bias}.txt"
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")

    bias_type = file.split("/")[-1].split(".")[1].lower()
    if not args.bias in bias_type: print("no",{args.bias})

    for line in open(file):
        split_line = line.strip().split("\t")
        context,question,choice = split_line[:3]
        label = split_line[3]
        stereotyped_groups = " ".join(split_line[4:])

        dataset.append(
            {   "context":context,
                "question": question,
                "choice":choice,
                "label": label,
                "bias": bias_type,
                STEREOTYPED_GROUPS: stereotyped_groups,
            }
        )
    return dataset


def load_realtoxicity():
    random.seed(24)
    data = [
        json.loads(line.strip())["prompt"]["text"]
        for line in open("data/realtoxicity.txt")
    ]
    return random.sample(data, 4000)[:2000]

def load_benchmark(args):
    if args.benchmark == "winogender":
        return load_winogender()
    elif args.benchmark == "bbq":
        return load_bbq(args)
    elif args.benchmark == "realtoxicity":
        return load_realtoxicity()
    elif args.benchmark == "bbq_debug":
        dataset = []
        for file in glob.glob("../data/bbq.sexual_orientation.txt"):
            bias_type = file.split("/")[-1].split(".")[1]

            for line in open(file):
                split_line = line.strip().split("\t")
                context,question,choice = split_line[:3]
                label = split_line[3]
                stereotyped_groups = " ".join(split_line[4:])

                dataset.append(
                {   "context":context,
                    "question": question,
                    "choice":choice,
                    "label": label,
                    "bias": bias_type,
                    STEREOTYPED_GROUPS: stereotyped_groups,
                }
            )
        random.shuffle(dataset)
        return dataset[:]


@torch.no_grad()
def prompting_bbq(args, tokenizer, llm, prompting_list):
    question_list = load_benchmark(args)
    return_list = []
    for question_ in tqdm(question_list,desc="bbq"):
        # print(question_)
        context, question, choice, label = question_["context"],question_["question"],question_["choice"], question_["label"]
        result_json_list = []
        history = ""
        for idx, prompt in enumerate(prompting_list):
            round = idx + 1
            query = copy.deepcopy(prompt)
            query = query.replace("#QUESTION", context+" "+question+" "+choice)
            input_query = copy.deepcopy(query)

            if round > 1:
                input_query = history + input_query

            response = get_response(args, tokenizer, llm, input_query)
            #print(f"output:{response}\tlabel:{label}")
            result_json = {
                "input": copy.deepcopy(input_query),
                "output": response,
                "round": round,
                "label": question_["label"],
                "bias": question_["bias"],
            }

            result_json_list.append(result_json)

            end_session = "</s>"
            history = history +" "+ input_query.strip() + " " + response + end_session

        return_list.append(result_json_list)
    return return_list


@torch.no_grad()
def prompting_winogender(args, tokenizer, llm, prompting_list):
    question_list = load_benchmark(args)
    return_list = []
    for question_ in tqdm(question_list, desc="winogender"):
        #print(question_)
        question, label = question_["question"], question_["label"]
        
        result_json_list = []
        history = ""
        for idx, prompt in enumerate(prompting_list):
            round = idx + 1
            query = copy.deepcopy(prompt)
            query = query.replace("#QUESTION", question)
            input_query = copy.deepcopy(query)

            if round > 1:
                input_query = history + input_query
            
            response = get_response(args, tokenizer, llm, input_query)
            #print(f"output:{response}\tlabel:{label}")
            result_json = {
                "input": copy.deepcopy(input_query),
                "output": response,
                "round": round,
                "label": label,
            }

            result_json_list.append(result_json)

            end_session = "</s>"
            history = history +" "+ input_query.strip() + " " + response + end_session

        return_list.append(result_json_list)
    return return_list

@torch.no_grad()
def prompting_toxicity(args, tokenizer, llm, prompting_list):
    question_list = load_benchmark(args)
    return_list = []
    for question in question_list:
        result_json_list = []
        history = ""
        for idx, prompt in enumerate(prompting_list):
            round = idx + 1
            query = copy.deepcopy(prompt)
            query = query.replace("#QUESTION", question)
            input_query = copy.deepcopy(query)

            if round > 1:
                input_query = history + input_query

            response = get_response(args, tokenizer, llm, input_query)
            #print(f"output:{response}\tlabel:{label}")
            print(idx)
            print(response)
            result_json = {
                "input": copy.deepcopy(input_query),
                "output": response,
                "round": round,
            }

            result_json_list.append(result_json)

            end_session = "</s>"
            history = history +" "+ input_query.strip() + " " + response + end_session

        return_list.append(result_json_list)
    return return_list

@torch.no_grad()
def get_response(args, tokenizer, llm, input_query, cot_round2=False):
    # print("input query:", input_query)
    input_ids = tokenizer(input_query, return_tensors="pt")
    max_new_tokens = args.max_new_tokens
    if cot_round2:
        if "bbq" in args.benchmark:
            max_new_tokens = 8
        elif "realtoxicity" in args.benchmark:
            if args.cot: 
                print("set realtoxicity")
                max_new_tokens = 100

    model_outputs = llm.generate(input_ids.input_ids.to(device), 
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id)
    # print("query length:", len(input_ids[0]))
    prompt_length = input_ids["input_ids"].shape[1]
    response = tokenizer.decode(
        model_outputs[0][prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = " ".join(copy.deepcopy(response.strip()).split("\n"))
    # print("response:", response)
    return response

@torch.no_grad()
def get_outputs(args, tokenizer, llm, input_query, max_new_tokens):
    # print("input query:", input_query)
    input_ids = tokenizer(input_query, return_tensors="pt")

    model_outputs = llm.generate(input_ids.input_ids.to(device), 
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.eos_token_id)
    # print("query length:", len(input_ids[0]))
    prompt_length = input_ids["input_ids"].shape[1]
    response = tokenizer.decode(
        model_outputs[0][prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = " ".join(copy.deepcopy(response.strip()).split("\n"))
    # print("response:", response)
    return response

class toxicityClf(nn.Module):
    def __init__(self, feature_dim):
        super(toxicityClf, self).__init__()
        self.linearProber = nn.Linear(feature_dim, 2).to(device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, features, labels):
        logits = self.linearProber(features).to(device)
        pred_softmax = torch.nn.functional.softmax(logits, dim=1)
        preds = np.argmax(pred_softmax.detach().cpu().numpy(), axis=1)
        loss = self.cross_entropy(logits, labels)

        return preds, loss.to(device)

def get_acc_qa(file):
    acc = 0
    with open(file,'r') as reader:
        data_json_list = json.load(reader)
        for data_json in data_json_list:
            for result in data_json:
                label, output = result["label"], result["output"]
                if label.lower() in output.lower(): acc += 1
        return acc / len(data_json_list)

def eva_toxicity(text):
    client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=PERSPECTIVE_API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    )

    analyze_request = {
    'comment': { 'text': text },
    'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
    # print(json.dumps(response, indent=2))