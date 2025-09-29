import config
import argparse

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from parse_string import Parser
from agents import AgentAction, HuggingfaceChatbot, BM25, EMB
from utils import *
import random
import numpy as np
import torch
from collections import Counter
import math
import re
# os.environ['HF_ENDPOINT'] = config.HF_ENDPOINT
os.environ['HF_TOKEN'] = config.HF_TOKEN
os.environ['HF_HOME'] = config.HF_HOME


    
def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)




def main(args):
    set_seeds(args)
    log(str(args),args.log_path)
    
    dataset = load_from_disk("/dataset")['train']
    questions = []
    for item in dataset:
        if item['type'] == 'question':
            questions.append(json.loads(item['content']))
            questions[-1]['source'] = item['source']


    if args.context:
        corpus_list = args.context.replace("_", " ").split('+')
    else:
        corpus_list = []
    
    source_list = []
    for item in dataset:
        if item['source'] not in source_list:
            source_list.append(item['source'])

    corpus = {}
    for source in source_list:
        corpus[source] = []
        for item in dataset:
            if item['source'] == source and item['type'] in corpus_list:
                corpus[source].append(item['content'])

    rag_list = {}
    for source in source_list:
        if corpus[source]:  
            if args.emb_search:
                rag_list[source] = EMB(corpus[source])
            else:
                rag_list[source] = BM25(corpus[source])
        else:
            rag_list[source] = None

    if args.api_name:
        chatbot = ''
    else:    
        chatbot = HuggingfaceChatbot(args.model)
        
    agents = AgentAction(chatbot, 
                         parser_fn = Parser().parse_answer,
                         template = args.prompt_template,
                         **vars(args))
    result_save_path = args.log_path.replace('.txt', '_results.txt')

    results = []

    true_list = []
    pred_list = []



    for i, question in enumerate(tqdm(questions)):
        if rag_list[question['source']]:
            question['context'] = [ctx.replace('\\', '').replace('(square)', '').replace('square)', '').replace('(square', '').replace(')square', '').replace('square(', '') for ctx in rag_list[question['source']].get_most_relevant(question['question'], num=args.hit)]
            question['context'] = '\n'.join(question['context'])
        else:
            question['context'] = ''
        question['question_content'] = question['question'] + '\nOptions: ' + str(question['options']) 
        label = question['correct_answer']
        true_list.append(label)
        decision = {}
        log(str(f"======== case: {i}\n"), args.log_path)
        for _ in range(args.generation_round):
            try:
                
                decision, message = agents.complete(**question)
                result = (decision["decision"] == label)
                results.append(result)
                acc = (sum(results) / len(results))
                print(acc)
                log(str(f"sample_id: {i} --- label: {label} --- result:{result} --- answer: {decision['decision']}\n"), args.log_path)
                decision['prompt'] = message
                log(str(decision)+"\n", args.log_path)
                if decision: 
                    pred_list.append(decision["decision"])
                    break

            except Exception as e:
                print(e)
                continue
        if not decision: 
            pred_list.append(-1)
            # results.append(-1)

    acc = (sum(results) / len(results))
    print(acc)
    y_true = pd.Series(true_list)
    y_pred = pd.Series(pred_list)
    y = pd.concat([y_true, y_pred], axis=1)
    torch.save(y, args.y_path)
    log(str(f"--- num_sample: {len(questions)} --- accuracy:{acc}\n"), args.log_path)
    log(str(f"--- num_sample: {len(questions)} --- accuracy:{acc}\n"), result_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--log_path", type=str, default="logs/try.txt")
    parser.add_argument("--prompt_template", type=str, default="prompts/answer_with_context.txt")
    parser.add_argument("--y_path", type=str, default="ys/try.pt")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--hit", type=int, default=3)
    parser.add_argument("--generation_round", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_name", type=str, default='')
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--api_model", type=str, default="")
    parser.add_argument("--api_token", type=str, default=config.api_key)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--emb_search", type=str, default="")
    args = parser.parse_args()
    main(args)
