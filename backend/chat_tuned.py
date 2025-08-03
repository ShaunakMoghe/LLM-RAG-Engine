from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# === Load base model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "microsoft/phi-2",
    max_seq_length = 512,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

# === Load LoRA adapter into the base model ===
model = PeftModel.from_pretrained(model, "fine_tuned_phi2", adapter_name="default")

# === Prompt ===
prompt = "<|user|>\nWhat armor set is used to defeat the Tier IV Voidgloom Seraph?\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# === Generate response ===
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    eos_token_id=tokenizer.eos_token_id,
)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Post-process output to trim multiple assistant generations ===
if "<|assistant|>" in decoded:
    # Take only the part after the *first* assistant token
    parts = decoded.split("<|assistant|>")
    if len(parts) > 1:
        response = parts[1].strip().split("<|")[0].strip()
    else:
        response = decoded.strip()
else:
    response = decoded.strip()

# === Show final output ===
print("\n=== Response ===\n")
print(response)
