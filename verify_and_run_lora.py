import json
import os
import sys
import random
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset
import seqeval.metrics
from peft import LoraConfig, TaskType, get_peft_model

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(16)

MODEL_NAME = "xlm-roberta-base"
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

SEEDS = [13, 42, 2024]
LR_LORA = 1e-3
EPOCHS_TRANS = 15
EPOCHS_SHOT = 15
BATCH_SIZE = 32

OUTPUT_DIR = "d:/NLP_PROMPT/experiments/results/fresh_verification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_indic_json_raw(file_path):
    items = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=128
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                raw_tag = label[word_idx]
                label_ids.append(LABEL2ID[raw_tag])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                filtered_p = []
                filtered_l = []
                for pred_id, label_id in zip(p, l):
                    if label_id != -100:
                        filtered_p.append(ID2LABEL[pred_id])
                        filtered_l.append(ID2LABEL[label_id])
                all_preds.append(filtered_p)
                all_labels.append(filtered_l)

    f1 = seqeval.metrics.f1_score(all_labels, all_preds, zero_division=0)
    prec = seqeval.metrics.precision_score(all_labels, all_preds, zero_division=0)
    rec = seqeval.metrics.recall_score(all_labels, all_preds, zero_division=0)
    acc = seqeval.metrics.accuracy_score(all_labels, all_preds)
    return float(acc), float(prec), float(rec), float(f1)

def get_lora_model(r=8):
    base_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"]
    )
    return get_peft_model(base_model, peft_config)

def main():
    print("=" * 80)
    print("TASK 1 & 2: VERIFYING TELUGU DATASETS (500-SHOT SUBSET & TEST SET OVERLAP)")
    print("=" * 80)

    te_train_all = load_indic_json_raw("d:/NLP_PROMPT/naamapadam_data/te/te_train.json")
    te_test_all = load_indic_json_raw("d:/NLP_PROMPT/naamapadam_data/te/te_test.json")

    te_500 = te_train_all[:500]

    train_tuples = [tuple(item["words"]) for item in te_500]
    unique_train_tuples = set(train_tuples)
    
    print(f"[DATASET CHECK] Total 500-shot Telugu training sentences loaded: {len(te_500)}")
    print(f"[DATASET CHECK] Unique sentence count in 500-shot subset: {len(unique_train_tuples)}")
    
    print("\n--- 5 SAMPLE SENTENCES WITH ENTITY LABELS FROM 500-SHOT SUBSET ---")
    for idx, item in enumerate(te_500[:5]):
        words = item["words"]
        tags = [t.strip() if isinstance(t, str) else t for t in item["ner"]]
        sentence_str = " ".join(words)
        entities = [(w, t) for w, t in zip(words, tags) if t != "O"]
        print(f"\nSample #{idx+1}:")
        print(f"  Sentence: {sentence_str}")
        print(f"  Word Count: {len(words)}")
        print(f"  Non-O Entities: {entities}")

    test_tuples = set(tuple(item["words"]) for item in te_test_all)
    overlap = unique_train_tuples.intersection(test_tuples)
    print(f"\n[LEAKAGE CHECK] Total test sentences in te_test.json: {len(te_test_all)}")
    print(f"[LEAKAGE CHECK] Train/Test Overlap Count: {len(overlap)}")
    if len(overlap) == 0:
        print("[LEAKAGE CHECK PASSED] Zero overlap detected between 500-shot Telugu training subset and Telugu test set!")
    else:
        print(f"[LEAKAGE CHECK FAILED] Found {len(overlap)} overlapping sentences!")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("TASK 3: DELETING PREVIOUS CACHED RESULTS & STARTING FRESH 3-SEED LORA-R8 RUN")
    print("=" * 80)

    cached_files = [
        "d:/NLP_PROMPT/experiments/results/fresh_verification/lora_r8_500shot_fresh.json",
        "d:/NLP_PROMPT/experiments/results/fresh_verification/lora_r8_500shot_fresh.csv",
        "d:/NLP_PROMPT/experiments/results/fresh_verification/loss_curve_seed13.json"
    ]
    for cf in cached_files:
        if os.path.exists(cf):
            os.remove(cf)
            print(f"[CLEANUP] Deleted prior cached file: {cf}")
        else:
            print(f"[CLEANUP] Confirmed absent: {cf}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Compute Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    hi_train_raw = load_indic_json_raw("d:/NLP_PROMPT/naamapadam_data/hi/hi_train.json")[:150]
    te_test_raw = te_test_all[:300]

    hi_train_ds = Dataset.from_dict({
        "tokens": [item["words"] for item in hi_train_raw],
        "ner_tags": [[t.strip() if isinstance(t, str) else t for t in item["ner"]] for item in hi_train_raw]
    }).map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=["tokens", "ner_tags"])

    te_500_ds = Dataset.from_dict({
        "tokens": [item["words"] for item in te_500],
        "ner_tags": [[t.strip() if isinstance(t, str) else t for t in item["ner"]] for item in te_500]
    }).map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=["tokens", "ner_tags"])

    te_test_ds = Dataset.from_dict({
        "tokens": [item["words"] for item in te_test_raw],
        "ner_tags": [[t.strip() if isinstance(t, str) else t for t in item["ner"]] for item in te_test_raw]
    }).map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=["tokens", "ner_tags"])

    hi_loader = DataLoader(hi_train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    te_loader = DataLoader(te_500_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(te_test_ds, batch_size=64, collate_fn=collator)

    seed_results = []
    loss_curves = {}

    for seed in SEEDS:
        print(f"\n>>> Running LoRA (r=8) for Seed {seed} <<<", flush=True)
        set_seed(seed)

        model = get_lora_model(r=8).to(device)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_LORA)

        # Source transfer on Hindi
        model.train()
        for epoch in range(1, EPOCHS_TRANS + 1):
            for batch in hi_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                outputs.loss.backward()
                optimizer.step()

        # Target Fine-Tuning on 500-shot Telugu with explicit loss logging
        model.train()
        epoch_losses = []
        print(f"--- 500-Shot Telugu Fine-Tuning Loss Curve (Seed {seed}) ---", flush=True)
        for epoch in range(1, EPOCHS_SHOT + 1):
            total_loss = 0.0
            num_batches = 0
            for batch in te_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch:02d}/{EPOCHS_SHOT:02d} | Train Loss: {avg_loss:.6f}", flush=True)

        loss_curves[seed] = epoch_losses

        # Evaluate on Telugu Test set
        acc, prec, rec, f1 = evaluate_model(model, test_loader, device)
        print(f"  [RESULT Seed {seed}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}", flush=True)

        seed_results.append({
            "Seed": seed,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

    # Task 4: Aggregation and Reporting
    res_df = pd.DataFrame(seed_results)
    
    mean_acc, std_acc = res_df["Accuracy"].mean(), res_df["Accuracy"].std()
    mean_prec, std_prec = res_df["Precision"].mean(), res_df["Precision"].std()
    mean_rec, std_rec = res_df["Recall"].mean(), res_df["Recall"].std()
    mean_f1, std_f1 = res_df["F1"].mean(), res_df["F1"].std()

    out_csv = "d:/NLP_PROMPT/experiments/results/fresh_verification/lora_r8_500shot_fresh.csv"
    out_json = "d:/NLP_PROMPT/experiments/results/fresh_verification/lora_r8_500shot_fresh.json"
    out_loss = "d:/NLP_PROMPT/experiments/results/fresh_verification/loss_curve_seed13.json"

    res_df.to_csv(out_csv, index=False)
    
    summary_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seeds": SEEDS,
        "metrics": {
            "Accuracy": f"{mean_acc:.4f} ± {std_acc:.4f}",
            "Precision": f"{mean_prec:.4f} ± {std_prec:.4f}",
            "Recall": f"{mean_rec:.4f} ± {std_rec:.4f}",
            "F1": f"{mean_f1:.4f} ± {std_f1:.4f}"
        },
        "raw_results": seed_results
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(out_loss, "w", encoding="utf-8") as f:
        json.dump({"seed_13_loss": loss_curves[13]}, f, indent=2)

    print("\n" + "=" * 80)
    print("TASK 4: FINAL CONFIRMED LORA (r=8) 500-SHOT RESULTS (MEAN ± STD)")
    print("=" * 80)
    print(f"Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Precision : {mean_prec:.4f} ± {std_prec:.4f}")
    print(f"Recall    : {mean_rec:.4f} ± {std_rec:.4f}")
    print(f"F1 Score  : {mean_f1:.4f} ± {std_f1:.4f}")

    print("\n--- FILE TIMESTAMPS PROVING FRESH RUN ---")
    for filepath in [out_csv, out_json, out_loss]:
        mtime = os.path.getmtime(filepath)
        dt_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"File: {filepath} | Created/Modified: {dt_str}")

if __name__ == "__main__":
    main()
