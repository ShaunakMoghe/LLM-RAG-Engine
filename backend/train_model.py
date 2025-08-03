from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import sys 


# === Load Base Model and Tokenizer ===
model_name = "microsoft/phi-2"
max_seq_length = 512
dtype = torch.bfloat16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

# === Setup LoRA for lightweight fine-tuning ===
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)


# === Load dataset path from command line ===
if len(sys.argv) < 2:
    raise ValueError("Missing dataset path argument. Usage: python train_model.py datasets/your_file.json")

dataset_path = sys.argv[1]
print(f"Using dataset: {dataset_path}")

# === Load and Format Dataset ===
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Add <|endoftext|> to explicitly mark where generation should stop
def format_prompt(example):
    return {
        "text": f"<|user|>\n{example['input']}\n<|assistant|>\n{example['output']}<|endoftext|>"
    }

dataset = dataset.map(format_prompt)
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

# === Training Configuration ===
training_args = TrainingArguments(
    output_dir="fine_tuned_phi2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=3e-4,
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

# === Trainer ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    dataset_text_field="text",
    args=training_args,
    max_seq_length=max_seq_length,
    packing=False,
)

trainer.train()
trainer.model.save_pretrained("fine_tuned_phi2")