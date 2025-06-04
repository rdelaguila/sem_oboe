from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "../data/corpus_ft/trex_ft_10k.jsonl"
OUT_MODEL = "../models/lora_trex"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, load_in_4bit=True, device_map="auto"
)

ds = load_dataset("json", data_files=DATA_PATH)["train"]

def preprocess(example):
    prompt = f"Extrae todas las ternas (Sujeto, Predicado, Objeto) del texto: {example['text']}\n"
    output = str(example['triplet_human'])
    return tokenizer(prompt + output, truncation=True, padding="max_length", max_length=256)

ds = ds.map(preprocess)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=OUT_MODEL,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    save_steps=500,
    logging_steps=100,
    fp16=False,
    save_total_limit=2,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained(OUT_MODEL)
tokenizer.save_pretrained(OUT_MODEL)
