import os
import sys
import subprocess
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import status

# === Unsloth and Torch Imports ===
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Global State ===
# We will load the model into these global variables later, instead of on startup
model = None
tokenizer = None
MODEL_PATH = "fine_tuned_phi2" # Define the path to your fine-tuned model

# === Upload Logic ===
UPLOAD_DIR = "datasets"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload", status_code=status.HTTP_200_OK)
async def upload_dataset(file: UploadFile, base_model: str = Form(...), task_type: str = Form(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"status": "success", "filename": file.filename}


# === Training Logic (Non-Blocking) ===
def run_training_in_background(dataset_path: str):
    """This function runs the training script in a background process."""
    global model, tokenizer
    print(f"🚀 Starting background training for {dataset_path}")
    
    # Invalidate the old model from memory
    model = None
    tokenizer = None
    
    result = subprocess.run([sys.executable, "train_model.py", dataset_path], capture_output=True, encoding="utf-8")
    if result.returncode == 0:
        print("✅ Training completed successfully.")
        print(result.stdout)
    else:
        print("❌ Training failed.")
        print(result.stderr)

class TrainRequest(BaseModel):
    filename: str

@app.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Triggers the training as a non-blocking background task."""
    dataset_path = os.path.join("datasets", request.filename)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    
    # Add the long-running training job to the background
    background_tasks.add_task(run_training_in_background, dataset_path)
    
    # Return a response to the user immediately
    return {"message": "Training has started in the background. Monitor the backend console for progress."}


# === Chat Logic (with On-Demand Model Loading) ===
def load_model():
    """Loads the fine-tuned model into the global variables."""
    global model, tokenizer
    print("🚀 Loading fine-tuned model into memory...")
    base_model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name="microsoft/phi-2",
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    tokenizer = _tokenizer
    model.eval().to("cuda")
    print("✅ Model loaded and ready!")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    global model, tokenizer
    
    # If model is not loaded, load it.
    if model is None or tokenizer is None:
        try:
            load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Model is not ready or failed to load. Please ensure training is complete.")

    prompt = f"<|user|>\n{request.message}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in decoded:
        parts = decoded.split("<|assistant|>")
        response = parts[1].strip().split("<|")[0].strip()
    else:
        response = decoded.strip()

    return {"response": response}