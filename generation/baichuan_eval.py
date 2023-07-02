'''
Author: Macielyoung
Date: 2023-06-07 09:51:59
Description: baichuan模型预测
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


base_model = "baichuan-inc/baichuan-7B"
# base_model = "../pretrained/baichuan-7b"  # 本地目录
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={"": 1}
)
model.eval()
model = model.to(device)
print("load model and tokenizer done!")


while True:
    text = input("please input your question:\n")
    input_ids = tokenizer(text, return_tensors='pt').input_ids
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
    output = results[0].strip()
    print("answer: {}\n".format(output))