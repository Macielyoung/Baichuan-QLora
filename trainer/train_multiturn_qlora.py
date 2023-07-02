'''
Author: Macielyoung
Date: 2023-06-17 15:33:54
Description: 模型QLora微调
'''
import os
import sys
from typing import List

import torch
import transformers
from datasets import load_dataset, load_from_disk
from loguru import logger

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer
import bitsandbytes as bnb
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class QLoraTrainer(Trainer):
    
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        logger.info("model save path: {}".format(output_dir))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)
        

def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "../dialogues",
    output_dir: str = "../models/",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    cutoff_len: int = 1024,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "query_key_value"
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    logger.info(
        f"Training Baichuan-QLoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='baichuan-inc/baichuan-7B'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
        
    # load model in 4bit
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=True,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # baichuan model without pad token
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))

    def tokenize_conversation(example):
        conversations = example['dialogue']
        
        input_ids = []
        labels = []
        for cid, conversation in enumerate(conversations):
            human_text = conversation['human']
            assistant_text = conversation['assistant']
            
            # 在前后位置添加开头和结束标记符号 (bos_token, eos_token)
            if cid == 0:
                human_text = tokenizer.bos_token + human_text + tokenizer.eos_token
            else:
                human_text = human_text + tokenizer.eos_token
            assistant_text += tokenizer.eos_token
            
            human_ids = tokenizer.encode(human_text)
            assistant_ids = tokenizer.encode(assistant_text)
            
            # 添加human id
            input_ids += human_ids
            labels += len(human_ids) * [-100]
            
            # 添加assistant id
            input_ids += assistant_ids
            labels += assistant_ids
            
        result = {
            'input_ids': input_ids,
            'labels': labels
        }
        return result
    
    
    def data_collator(features: list) -> dict:
        # cut off the input and label
        input_ids_list = [feature['input_ids'][:cutoff_len] for feature in features]
        labels_list = [feature['labels'][:cutoff_len] for feature in features]
        # logger.info("input_ids_list: {}".format(input_ids_list))
        # logger.info("labels_list: {}".format(labels_list))
        # logger.info("input shape: {}, {}".format(len(input_ids_list), len(input_ids_list[0])))
        
        # pad token from left
        input_ids = pad_sequence([torch.tensor(input_ids[::-1]) for input_ids in input_ids_list], 
                                 batch_first=True,
                                 padding_value=tokenizer.pad_token_id).flip(dims=[1])
        labels = pad_sequence([torch.tensor(labels[::-1]) for labels in labels_list], 
                              batch_first=True, 
                              padding_value=-100).flip(dims=[1])
        
        input_ids = input_ids.long()
        labels = labels.long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id),
            "labels": labels,
        }

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
    
    # add adapter modules for all linear layer
    lora_target_modules = find_all_linear_names(model)
    logger.info("lora target modules: {}".format(lora_target_modules))
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".jsonl"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_from_disk(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            logger.info(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    model.config.torch_dtype = torch.float32
    
    logger.info("data info: {}".format(data))
    if val_set_size > 0:
        # split data into train and test dataset
        train_val = data['train'].train_test_split(
            test_size=val_set_size, shuffle=True, seed=215
        )
        train_data = (
            train_val['train'].shuffle().map(tokenize_conversation, remove_columns=train_val['train'].column_names)
        )
        val_data = (
            train_val['test'].shuffle().map(tokenize_conversation, remove_columns=train_val['test'].column_names)
        )
    else:
        train_data = data['train'].shuffle().map(tokenize_conversation, remove_columns=data['train'].column_names)
        val_data = data['test'].shuffle().map(tokenize_conversation, remove_columns=data['test'].column_names)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        
    train_args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=200,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=50,
        optim="adamw_torch",
        # optim="paged_adamw_32bit",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if val_set_size > 0 else None,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )

    trainer = QLoraTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=train_args,
        data_collator=data_collator,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    logger.info(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="../dialogues", help="sft data path")
    parser.add_argument("--pretrained", type=str, default="baichuan-inc/baichuan-7B", help="pretrained model from huggingface hub")
    parser.add_argument("--save_path", type=str, default="../models", help="model saved path")
    parser.add_argument("--epoches", type=int, default=5, help="epoch num")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="micro batch size")
    parser.add_argument("--val_set_size", type=int, default=2000, help="validation set size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate num")
    parser.add_argument("--max_length", type=int, default=2048, help="sentence max length")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    # args = parse_args()
    # train(base_model=args.pretrained,
    #       data_path=args.data_path,
    #       output_dir=args.save_path,
    #       batch_size=args.batch_size,
    #       micro_batch_size=args.micro_batch_size,
    #       num_epochs=args.epoches,
    #       learning_rate=args.lr,
    #       cutoff_len=args.max_length,
    #       val_set_size=args.val_set_size,
    #       lora_r=args.lora_r,
    #       lora_alpha=args.lora_alpha,
    #       lora_dropout=args.lora_dropout,
    #       )
    base_model = "baichuan-inc/baichuan-7B"
    train(base_model=base_model)
    
