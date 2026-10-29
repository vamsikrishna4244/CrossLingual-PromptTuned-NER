import os
import json
import re
import sys
from datasets import load_dataset

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')

output_path = "d:/NLP_PROMPT/experiments/kokborok_hf_augmented.json"
os.makedirs("d:/NLP_PROMPT/experiments", exist_ok=True)

print("=== Fetching & Processing Kokborok Dataset from Hugging Face ===")

sentences = []

# 1. Fetch agnivamaiti/kokborok-qa
try:
    print("Loading dataset 'agnivamaiti/kokborok-qa' from Hugging Face...")
    ds_qa = load_dataset("agnivamaiti/kokborok-qa")
    for r in ds_qa['train']:
        p = r.get('prompt', '').replace('User:', '').strip()
        c = r.get('completion', '').replace('Assistant:', '').strip()
        if p and len(p.split()) >= 3:
            sentences.append(p)
        if c and len(c.split()) >= 3:
            sentences.append(c)
    print(f"Extracted {len(sentences)} sentence candidates from agnivamaiti/kokborok-qa.")
except Exception as e:
    print(f"Could not fetch agnivamaiti/kokborok-qa: {e}")

# 2. Process and Clean Sentences
cleaned_sentences = []
for s in sentences:
    # Basic cleaning
    s_clean = re.sub(r'\s+', ' ', s).strip()
    if 4 <= len(s_clean.split()) <= 40:
        cleaned_sentences.append(s_clean)

unique_sentences = sorted(list(set(cleaned_sentences)))
print(f"Total Unique Filtered Kokborok Sentences: {len(unique_sentences)}")

# 3. Apply Weak Annotation Rules for NER Training Dataset Construction
from app.app import analyze_ner_sentence

annotated_dataset = []
entity_sentence_count = 0

for idx, text in enumerate(unique_sentences):
    res = analyze_ner_sentence(text, model_name="kokborok_prompt_tuned")
    tokens = res.get("tokens", [])
    tags = res.get("tags", [])
    has_entity = any(t != "O" for t in tags)
    if has_entity:
        entity_sentence_count += 1
    
    annotated_dataset.append({
        "id": f"hf_kokborok_{idx}",
        "words": tokens,
        "ner": tags,
        "text": text,
        "has_entity": has_entity
    })

print(f"Annotated Dataset Stats:")
print(f" - Total Kokborok Sentences: {len(annotated_dataset)}")
print(f" - Sentences with Named Entities (PER/LOC/ORG): {entity_sentence_count}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(annotated_dataset, f, ensure_ascii=False, indent=2)

print(f"Successfully saved Hugging Face Kokborok dataset to: {output_path}")
