from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Path to your local model
model_path = "./chatbot/checkpoint-450"

# ✅ Load model from safetensors
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model/tokenizer: {e}")

# ✅ Set up FastAPI
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],           # already OK
    allow_headers=["*"],           # already OK
    allow_origin_regex=".*",       # fallback to allow any origin if needed
)

# ✅ Request schema
class ChatRequest(BaseModel):
    user_input: str
    context: str = "You are a helpful agriculture assistant."

# ✅ Generate response
def generate_response(user_input: str, context: str):
    prompt = f"{context}\n\nUser: {user_input}\n\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    output = model.generate(
        inputs["input_ids"],
        max_length=150,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=2
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Bot:")[-1].strip() if "Bot:" in decoded else decoded.strip()

# ✅ POST endpoint
@app.post("/server")
async def chat(request: ChatRequest):
    print(request)
    try:
        result = generate_response(request.user_input, request.context)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")
