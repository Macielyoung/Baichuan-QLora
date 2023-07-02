'''
Author: Macielyoung
Date: 2023-06-17 14:33:54
Description: 数据预处理
'''
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
import json
import random
from loguru import logger


MAX_SENTENCE_LENGTH = 1000


def read_moss_file(data_file):
    conversations = []
    with open(data_file, 'r') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            # print(json_line)
            # break
            conversation = json_line['conversation']
            dialogue = ""
            for turn in conversation:
                human = turn['human']
                assistant = turn['assistant']
                dialogue += human + assistant
            
            if len(dialogue) > MAX_SENTENCE_LENGTH:
                add_conversation = [conversation[0]]
            else:
                add_conversation = conversation
            conversation_json = json.dumps(add_conversation, ensure_ascii=False)
            conversations.append(conversation_json)
        return conversations
    
    
def read_ultrachat_file(data_file):
    conversations = []
    with open(data_file, 'r') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            # print(json_line)
            # break
            conversation = json_line['conversation']
            dialogue = ""
            for turn in conversation:
                human = turn['human']
                assistant = turn['assistant']
                dialogue += human + " " + assistant
            
            if len(dialogue.split(" ")) > MAX_SENTENCE_LENGTH:
                add_conversation = [conversation[0]]
            else:
                add_conversation = conversation
            conversation_json = json.dumps(add_conversation, ensure_ascii=False)
            conversations.append(conversation_json)
            # print(conversation_json)
        return conversations
    


def process_moss_dataset(example):
    '''处理moss sft3数据集'''
    res = defaultdict(list)
    conversations = example['conversation']
    
    for conversation in conversations:
        dialogue = ""
        for turn in conversation:
            human = turn['human']
            assistant = turn['assistant']
            dialogue += human + assistant
        
        # res['cid'].append(cid)
        if len(dialogue) > MAX_SENTENCE_LENGTH:
            res['dialogue'].append([conversation[0]])
        else:
            res['dialogue'].append(conversation)
    return res


def process_ultrachat_dataset(example):
    '''处理ultrachat数据集'''
    res = defaultdict(list)
    conversations = example['conversation']
    
    for conversation in conversations:
        dialogue = ""
        for turn in conversation:
            human = turn['human']
            assistant = turn['assistant']
            dialogue += human + " " + assistant
        
        # res['cid'].append(cid)
        if len(dialogue.split(" ")) > MAX_SENTENCE_LENGTH:
            res['dialogue'].append([conversation[0]])
        else:
            res['dialogue'].append(conversation)
    return res
    
    
if __name__ == "__main__":
    # moss sft3 dataset: https://huggingface.co/datasets/YeungNLP/moss-003-sft-data
    moss_file = "../datasets/moss-003-sft-data.jsonl"
    moss_dataset = load_dataset("json", data_files=moss_file)['train']
    moss_dataset = moss_dataset.map(process_moss_dataset,
                                    batched=True,
                                    batch_size=20,
                                    num_proc=10,
                                    remove_columns=moss_dataset.column_names)
    logger.info("load moss data done, info: {}".format(moss_dataset))
    logger.info("moss first line: {}".format(moss_dataset['dialogue'][0]))
    
    # ultrachat dataset: https://huggingface.co/datasets/YeungNLP/ultrachat
    ultrachat_file = "../datasets/ultra-chat.jsonl"
    ultrachat_dataset = load_dataset("json", data_files=ultrachat_file)['train']
    ultrachat_dataset = ultrachat_dataset.map(process_ultrachat_dataset,
                                              batched=True,
                                              batch_size=20,
                                              num_proc=10,
                                              remove_columns=ultrachat_dataset.column_names)
    logger.info("load ultrachat data done, info: {}".format(ultrachat_dataset))
    logger.info("ultrachat first line: {}".format(ultrachat_dataset['dialogue'][0]))
    
    dialogue_dataset = concatenate_datasets([moss_dataset, ultrachat_dataset])
    
    dataset_num = len(dialogue_dataset)
    sample_num = 500000
    sample_list = random.sample(range(dataset_num), sample_num)
    dialogue_dataset = dialogue_dataset.select(sample_list)
    logger.info("dialogue dataset info: {}".format(dialogue_dataset))
    logger.info("dialogue data first line: {}".format(dialogue_dataset['dialogue'][0]))
    
    train_val_dataset = dialogue_dataset.train_test_split(
        test_size=5000, shuffle=True, seed=215
    )
    logger.info("train and eval dataset info: {}".format(train_val_dataset))
    
    dialogue_path = "../dialogues"
    train_val_dataset.save_to_disk(dialogue_path)