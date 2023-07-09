'''
Author: Macielyoung
Date: 2023-06-07 09:51:59
Description: QLora模型预测
'''
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import readline


# base_model = "baichuan-inc/baichuan-7B"
base_model = "../pretrained/baichuan-7b"  # 本地目录
qlora_model = "../models/checkpoint-2000"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={"": 0}
)
model = PeftModel.from_pretrained(model, qlora_model)
model.eval()
model = model.to(device)
print("load model and tokenizer done!")

history = []

while True:
    text = input("please input your question:\n")
    if len(history) == 0:
        prompt = "{}{}{}".format(tokenizer.bos_token, text, tokenizer.eos_token)
    else:
        prompt = ""
        for pid, pair in enumerate(history):
            human = pair['human']
            assistant = pair['assistant']
            if pid == 0:
                prompt += tokenizer.bos_token
            prompt += human + tokenizer.eos_token + assistant + tokenizer.eos_token
        prompt += text + tokenizer.eos_token
    print("prompt: {}\n\n".format(prompt))     
            
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    input_ids = input_ids.to(device)
    # print("input_ids: {}".format(input_ids))
    outputs = model.generate(input_ids=input_ids,
                             max_new_tokens=1024,
                             do_sample=True, 
                             top_p=0.75,
                             temperature=0.5,
                             repetition_penalty=1.2, 
                             eos_token_id=tokenizer.eos_token_id)
    outputs = outputs[:, input_ids.shape[-1]: ]
    results = tokenizer.batch_decode(outputs)
    # print("output tokens: {}".format(outputs))
    response = results[0].strip()
    print("answer: {}\n".format(response))
    
    history.append({'human': text, 'assistant': response})