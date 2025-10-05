import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, LoraModel, get_peft_model
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import Trainer, TrainingArguments, TrainerCallback
import json
import numpy as np
import sys

class TrainerSaveLossCallback(TrainerCallback):
    def __init__(self, output_dir, output_file = "losses.json"):
        self.output_dir = output_dir
        self.output_file = output_file
        self.loss_data = {"train": [], "eval": []}

    def on_log(self, args, state, control, logs = None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.loss_data["train"].append({"step": state.global_step, "loss": logs["loss"]})
            if "eval_loss" in logs:
                self.loss_data["eval"].append({"step": state.global_step, "eval_loss": logs["eval_loss"]})

    def on_train_end(self, args, state, control, **kwargs):
        p = f"{self.output_dir}-{self.output_file}"
        with open(p, "w") as f:
            json.dump(self.loss_data, f)
        print(f"Losses saved to {p}")


save_path = f"results1"
model_path = "vinai/PhoGPT-4B-Chat"

config = AutoConfig.from_pretrained(model_path, trust_remote_code = True)  
config.attn_config['attn_impl'] = 'torch'
model = AutoModelForCausalLM.from_pretrained(model_path, config = config, torch_dtype = torch.bfloat16, trust_remote_code=  True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)  

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 8,
    target_modules = [
        "Wqkv",
    ],
    lora_dropout = 0.05,
    bias = "none",
    use_rslora = True,
)
model = get_peft_model(model, lora_config)


# dataset
def pad_t(t, max_len, pad_token_id):
    length = t.shape[0]

    if length == max_len:
        return t
    elif length > max_len:
        return t[:max_len:]
    else:
        pad_len = max_len - length
        pad = torch.full((pad_len,), pad_token_id, dtype = t.dtype, device = t.device)
        return torch.cat([t, pad], dim = 0)
    
def get_q_a(d):
    q = d["title"].strip()
    a = d["content"].strip()
    
    if q == "" or a == "":
        return q, a
    
    a = a.split(q)[1][1:]
    a = a.split(". ")

    return q, ". ".join(a)
    

PROMPT_TEMPLATE = "### Câu hỏi: {q}\n### Trả lời:"  

class MyDataset(Dataset):
    def __init__(self, df, max_len):
        self.data = df.to_dict(orient = 'records')
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        quest, ans = get_q_a(self.data[index])
        
        input_prompt = PROMPT_TEMPLATE.format(q = quest)  
        inp = tokenizer(text = input_prompt, return_tensors = "pt", padding = False, truncation = False)
        full = tokenizer(text = input_prompt + " " + ans, return_tensors = "pt", padding = False, truncation = False)
        
        inp = {k: v.squeeze(0) for k, v in inp.items()}
        full = {k: v.squeeze(0) for k, v in full.items()}
        
        full["labels"] = full["input_ids"].clone()
        full["labels"][:inp["input_ids"].shape[0]:] = -100
        
        full["input_ids"] = pad_t(full["input_ids"], self.max_len, tokenizer.pad_token_id)
        full["attention_mask"] = pad_t(full["attention_mask"], self.max_len, 0)
        full["labels"] = pad_t(full["labels"], self.max_len, -100)

        return full
    
df = pd.read_csv("train.csv")

df["cplx"] = df["content"].apply(lambda s: len(s.split(". ")))

df["title"] = df["title"].fillna("")
df["content"] = df["content"].fillna("")

df = df[df["cplx"] < 8]

ds = MyDataset(df, 512)

training_args = TrainingArguments(
    output_dir = save_path,
    
    num_train_epochs = 1,
    learning_rate = 3e-5,
    
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,

    gradient_accumulation_steps = 2,
    eval_accumulation_steps = 1,
    
    eval_strategy = "no",
    save_strategy = "no",

    
    logging_strategy = "steps",
    logging_steps = 100,
    
    lr_scheduler_type = "linear",
    warmup_steps = 100,   
    
    label_names = ["labels"],
    
    bf16 = True,
    
    report_to = "none",
    
    dataloader_num_workers = 16,
    dataloader_persistent_workers = True,
    dataloader_pin_memory = True,

    logging_first_step = True,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = ds,
    eval_dataset = ds,
    tokenizer = tokenizer,
    callbacks = [TrainerSaveLossCallback(save_path)]
)

trainer.train()
trainer.save_model(f"best_phase_1")