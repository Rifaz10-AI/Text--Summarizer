from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

templates = Jinja2Templates(directory="templates")  # ✅ use subfolder

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip()  # ✅ removed .lower()

def summarize_dialogue(dialogue: str) -> str:
    dialogue = "summarize: " + clean_data(dialogue)  # ✅ T5 task prefix

    inputs = tokenizer(
        dialogue,
        max_length=512,          # ✅ removed padding="max_length"
        truncation=True,
        return_tensors="pt"
    ).to(device)

    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(targets[0], skip_special_tokens=True)

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    if not dialogue_input.dialogue.strip():           # ✅ empty check
        return {"error": "Input dialogue cannot be empty"}
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")