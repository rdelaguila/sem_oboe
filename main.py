from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/triples_ft/trex_ft_10k.json"
OUT_MODEL = "models/lora_trex"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto"
)

ds = load_dataset("json", data_files=DATA_PATH)["train"]
print(ds["train"])  # Muestra los campos disponibles
