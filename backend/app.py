import streamlit as st
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

st.set_page_config(page_title="AI Chatter", page_icon="🛡️", layout="centered")
st.title("Chatbot")
st.subheader("Ask a Question")

user_input = st.text_input(" ", placeholder="Question?")

if user_input:
    with st.spinner("Thinking..."):
        # Load model + adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="microsoft/phi-2",
            max_seq_length=512,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, "fine_tuned_phi2", adapter_name="default")

        # Format prompt
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Show result
        st.markdown("### Answer:")
        st.markdown(response)