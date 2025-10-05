import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, LoraModel, get_peft_model
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
import json

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


save_path = "results"
model_path = "vinai/PhoGPT-4B-Chat"  

config = AutoConfig.from_pretrained(model_path, trust_remote_code = True)  
# config.attn_config['attn_impl'] = 'torch'
config.attn_implementation = "flash_attention_2"
model = AutoModelForCausalLM.from_pretrained(model_path, config = config, torch_dtype = torch.bfloat16, trust_remote_code=  True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)  

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    target_modules = [
        "Wqkv",
        "up_proj",
        "down_proj",
    ],
    lora_dropout = 0.05,
    bias = "none",
    use_rslora = True,
)
model = get_peft_model(model, lora_config)


# dataset
PROMPT_TEMPLATE = "### Câu hỏi: {q}\n### Trả lời:"  

class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df.to_dict(orient = 'records')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        q = self.data[index]["title"]
        a = self.data[index]["content"]
        input_prompt = PROMPT_TEMPLATE.format(q = q)  
        inp = tokenizer(text = input_prompt, text_target = a, return_tensors = "pt", padding = "max_length", max_length = 1024, truncation = True)
        inp = {k: v.squeeze(0) for k, v in inp.items()}
        return inp
    
df = pd.read_csv("train.csv")
train_df, eval_df = train_test_split(df, test_size = 0.1, random_state = 22022009)

train_ds = MyDataset(train_df)
eval_ds = MyDataset(eval_df)

training_args = TrainingArguments(
    output_dir = save_path,
    
    num_train_epochs = 3,
    learning_rate = 5e-5,
    
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 8,

    gradient_accumulation_steps = 4,
    eval_accumulation_steps = 1,
    
    eval_strategy = "steps",
    eval_steps = 1,
    
    save_strategy = "steps",
    save_steps = 1,
    metric_for_best_model = "eval_loss",

    save_total_limit = 3,
    
    logging_strategy = "steps",
    logging_steps = 100,
    
    lr_scheduler_type = "linear",
    warmup_steps = 100,   
    
    label_names = ["labels"],
    
    bf16 = True,
    
    report_to = "none",
    
    dataloader_num_workers = 4,
    dataloader_persistent_workers = True,
    dataloader_pin_memory = True,

    logging_first_step = True,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = eval_ds,
    tokenizer = tokenizer,
    callbacks = [TrainerSaveLossCallback(save_path)]
)

trainer.train()
trainer.save_model("best")