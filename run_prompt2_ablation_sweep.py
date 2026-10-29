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
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, load_dataset
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
LR_DEFAULT = 1e-3
BATCH_SIZE = 32

OUTPUT_DIR = "d:/NLP_PROMPT/experiments/results/prompt2_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_indic_json_raw(file_path, max_samples=None):
    items = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            if line.strip():
                items.append(json.loads(line))
    return items

def load_english_conll(max_samples=150):
    try:
        ds_en = load_dataset("lhoestq/conll2003", split=f"train[:{max_samples}]")
        raw_tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        target_set = set(LABEL_LIST)
        items = []
        for sample in ds_en:
            words = sample['tokens']
            mapped_ner = [raw_tags[t] if raw_tags[t] in target_set else 'O' for t in sample['ner_tags']]
            items.append({"words": words, "ner": mapped_ner})
        return items
    except Exception as e:
        print(f"[WARNING] Could not load English CoNLL dataset: {e}")
        return []

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=64
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

def get_lora_model(r=8, target_modules=["query", "value"]):
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
        target_modules=target_modules,
        modules_to_save=["classifier"]
    )
    return get_peft_model(base_model, peft_config)

def run_experiment(exp_name, exp_config, tokenizer, collator, datasets_dict, device):
    print(f"\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Config: {exp_config}")
    print("=" * 80)

    target_modules = exp_config.get("target_modules", ["query", "value"])
    use_warmup = exp_config.get("use_warmup", False)
    target_epochs = exp_config.get("target_epochs", 15)
    use_class_weights = exp_config.get("use_class_weights", False)
    source_langs = exp_config.get("source_langs", ["hi"])

    # Load source training data based on source_langs
    source_raw = []
    if "hi" in source_langs:
        source_raw.extend(datasets_dict["hi_train"])
    if "ta" in source_langs:
        source_raw.extend(datasets_dict["ta_train"])
    if "ml" in source_langs:
        source_raw.extend(datasets_dict["ml_train"])
    if "en" in source_langs:
        source_raw.extend(datasets_dict["en_train"])

    source_ds = Dataset.from_dict({
        "tokens": [item["words"] for item in source_raw],
        "ner_tags": [[t.strip() if isinstance(t, str) else t for t in item["ner"]] for item in source_raw]
    }).map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=["tokens", "ner_tags"])

    source_loader = DataLoader(source_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    te_500_loader = DataLoader(datasets_dict["te_500_ds"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(datasets_dict["te_val_ds"], batch_size=64, collate_fn=collator)

    # Class weighting setup
    class_weights_tensor = None
    if use_class_weights:
        # Downweight 'O' class (index 0) to 0.3 and upweight entity classes to 2.0
        weights = [0.3] + [2.0] * (NUM_LABELS - 1)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    seed_metrics = []

    for seed in SEEDS:
        set_seed(seed)
        model = get_lora_model(r=8, target_modules=target_modules).to(device)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_DEFAULT)

        # Phase 1: Source Transfer Pre-training
        model.train()
        for epoch in range(1, 15 + 1):
            for batch in source_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                outputs.loss.backward()
                optimizer.step()

        # Phase 2: Target 500-Shot Fine-Tuning
        total_steps = len(te_500_loader) * target_epochs
        scheduler = None
        if use_warmup:
            warmup_steps = int(0.10 * total_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        model.train()
        for epoch in range(1, target_epochs + 1):
            for batch in te_500_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if use_class_weights:
                    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=-100)
                    loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    loss = outputs.loss

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        # Evaluate on Held-out VALIDATION split
        acc, prec, rec, f1 = evaluate_model(model, val_loader, device)
        print(f"  [Seed {seed}] Validation Metrics | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Acc: {acc:.4f}")

        seed_metrics.append({
            "Seed": seed,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

    res_df = pd.DataFrame(seed_metrics)
    mean_f1, std_f1 = res_df["F1"].mean(), res_df["F1"].std()
    mean_prec, std_prec = res_df["Precision"].mean(), res_df["Precision"].std()
    mean_rec, std_rec = res_df["Recall"].mean(), res_df["Recall"].std()
    mean_acc, std_acc = res_df["Accuracy"].mean(), res_df["Accuracy"].std()

    print(f"\n--> {exp_name} VALIDATION SUMMARY across seeds:")
    print(f"    F1 Score  : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"    Precision : {mean_prec:.4f} ± {std_prec:.4f}")
    print(f"    Recall    : {mean_rec:.4f} ± {std_rec:.4f}")
    print(f"    Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}")

    return {
        "Experiment": exp_name,
        "Validation_F1_Mean": mean_f1,
        "Validation_F1_Std": std_f1,
        "Validation_F1_Formatted": f"{mean_f1:.4f} ± {std_f1:.4f}",
        "Validation_Precision": f"{mean_prec:.4f} ± {std_prec:.4f}",
        "Validation_Recall": f"{mean_rec:.4f} ± {std_rec:.4f}",
        "Validation_Accuracy": f"{mean_acc:.4f} ± {std_acc:.4f}"
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Compute Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Load datasets
    hi_train_raw = load_indic_json_raw("d:/NLP_PROMPT/naamapadam_data/hi/hi_train.json", max_samples=150)
    ta_train_raw = load_indic_json_raw("d:/NLP_PROMPT/naamapadam_data/ta/ta_train.json", max_samples=150)
    ml_train_raw = load_indic_json_raw("d:/NLP_PROMPT/naamapadam_data/ml/ml_train.json", max_samples=150)
    en_train_raw = load_english_conll(max_samples=150)

    te_500_raw = load_indic_json_raw("d:/NLP_PROMPT/naamapadam_data/te/te_train.json", max_samples=500)
    te_val_raw = load_indic_json_raw("d:/NLP_PROMPT/naamapadam_data/te/te_val.json", max_samples=300)

    te_500_ds = Dataset.from_dict({
        "tokens": [item["words"] for item in te_500_raw],
        "ner_tags": [[t.strip() if isinstance(t, str) else t for t in item["ner"]] for item in te_500_raw]
    }).map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=["tokens", "ner_tags"])

    te_val_ds = Dataset.from_dict({
        "tokens": [item["words"] for item in te_val_raw],
        "ner_tags": [[t.strip() if isinstance(t, str) else t for t in item["ner"]] for item in te_val_raw]
    }).map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=["tokens", "ner_tags"])

    datasets_dict = {
        "hi_train": hi_train_raw,
        "ta_train": ta_train_raw,
        "ml_train": ml_train_raw,
        "en_train": en_train_raw,
        "te_500_ds": te_500_ds,
        "te_val_ds": te_val_ds
    }

    experiments = [
        ("Baseline (LoRA r=8, q+v, Const LR, 15 Epochs, Hi+Te)", {
            "target_modules": ["query", "value"],
            "use_warmup": False,
            "target_epochs": 15,
            "use_class_weights": False,
            "source_langs": ["hi"]
        }),
        ("Task 1: Target Modules Expansion [q, k, v, dense]", {
            "target_modules": ["query", "key", "value", "dense"],
            "use_warmup": False,
            "target_epochs": 15,
            "use_class_weights": False,
            "source_langs": ["hi"]
        }),
        ("Task 2: LR Warmup (10% linear warmup + decay)", {
            "target_modules": ["query", "value"],
            "use_warmup": True,
            "target_epochs": 15,
            "use_class_weights": False,
            "source_langs": ["hi"]
        }),
        ("Task 3: Increased Target Epochs (35 Epochs)", {
            "target_modules": ["query", "value"],
            "use_warmup": False,
            "target_epochs": 35,
            "use_class_weights": False,
            "source_langs": ["hi"]
        }),
        ("Task 4: Class-Weighted Cross-Entropy Loss", {
            "target_modules": ["query", "value"],
            "use_warmup": False,
            "target_epochs": 15,
            "use_class_weights": True,
            "source_langs": ["hi"]
        }),
        ("Task 5: Multilingual Source Transfer (En+Hi+Ta+Ml)", {
            "target_modules": ["query", "value"],
            "use_warmup": False,
            "target_epochs": 15,
            "use_class_weights": False,
            "source_langs": ["hi", "ta", "ml", "en"] if len(en_train_raw) > 0 else ["hi", "ta", "ml"]
        })
    ]

    all_exp_results = []

    for exp_name, exp_cfg in experiments:
        res = run_experiment(exp_name, exp_cfg, tokenizer, collator, datasets_dict, device)
        all_exp_results.append(res)

    res_df = pd.DataFrame(all_exp_results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "prompt2_ablation_results.csv"), index=False)
    
    with open(os.path.join(OUTPUT_DIR, "prompt2_ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_exp_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PROMPT 2 INDIVIDUAL ABLATION RESULTS SUMMARY (EVALUATED ON HELD-OUT VALIDATION SPLIT)")
    print("=" * 80)
    for res in all_exp_results:
        print(f"{res['Experiment']:<55} | Val F1: {res['Validation_F1_Formatted']}")

if __name__ == "__main__":
    main()
