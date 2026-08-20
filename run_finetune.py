"""
Headless runner for the GPT-2 IMDb sentiment fine-tuning assignment.
Mirrors finetune_gpt2_imdb.ipynb steps 1-12 (skips the blocking Gradio launch).

Env vars (defaults = assignment spec):
  N_TRAIN   (default 3000)
  N_EVAL    (default 1000)
  EPOCHS    (default 2)
  MAX_LEN   (default 512)
"""
import os
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

N_TRAIN = int(os.environ.get("N_TRAIN", 3000))
N_EVAL = int(os.environ.get("N_EVAL", 1000))
EPOCHS = float(os.environ.get("EPOCHS", 2))
MAX_LEN = int(os.environ.get("MAX_LEN", 512))
LR = float(os.environ.get("LR", 2e-5))

# Step 1 - hardware
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"[1] torch {torch.__version__} | device = {device}", flush=True)

# Step 2 - dataset
dataset = load_dataset("stanfordnlp/imdb")
print(f"[2] IMDb loaded: {dataset['train'].num_rows} train / {dataset['test'].num_rows} test", flush=True)

# Step 3 - tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print(f"[3] tokenizer ready, pad_token = {tokenizer.pad_token}", flush=True)

# Step 4 - tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

tokenized = dataset.map(tokenize_function, batched=True)
print("[4] tokenized", flush=True)

# Step 5 - subsets
small_train = tokenized["train"].shuffle(seed=42).select(range(N_TRAIN))
small_eval = tokenized["test"].shuffle(seed=42).select(range(N_EVAL))
print(f"[5] subsets: {len(small_train)} train / {len(small_eval)} eval", flush=True)

# Step 6 - model
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)
model.config.pad_token_id = tokenizer.pad_token_id
print("[6] model + 2-class head ready", flush=True)

# Step 7 - metric
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Step 8 - training args
training_args = TrainingArguments(
    output_dir="./gpt2-imdb-finetuned",
    learning_rate=LR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
)
print("[8] training args set", flush=True)

# Step 9 - train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
print("[9] starting training...", flush=True)
trainer.train()

# Step 10 - evaluate
eval_results = trainer.evaluate()
print(f"[10] FINAL EVAL: {eval_results}", flush=True)

# Step 11 - save
save_dir = "./gpt2-imdb-sentiment"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"[11] saved to {save_dir}", flush=True)

# Step 12 - inference pipeline sanity check
clf = pipeline("sentiment-analysis", model=save_dir, tokenizer=save_dir,
               device=0 if torch.cuda.is_available() else -1, truncation=True, max_length=MAX_LEN)
for t in ["An absolute masterpiece, I loved every minute of it!",
          "A boring, predictable mess. Total waste of time."]:
    print(f"[12] {clf(t)}  <- {t[:40]}", flush=True)

print("=== PIPELINE COMPLETE ===", flush=True)
