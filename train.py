import argparse
import ast
import json
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

# Metrics
metric = evaluate.load("seqeval")

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value
    return final_results

def get_label_list(dataset):
    label_set = set()
    for data in dataset:
        labels = data["ner_tags"]  # Adjust this field name based on your dataset structure
        label_set.update(labels)
    return list(label_set)

def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
    )
    labels = []
    for i, example_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        last_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != last_word_id:
                label_id = label_to_id.get(example_labels[word_id], -100)
                label_ids.append(label_id)
            else:
                label_ids.append(-100)  # or label_ids.append(label_id) if you want to label sub-tokens
            last_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main(args):
    # Load dataset
    raw_datasets = load_dataset(args.dataset_name)

    raw_datasets = raw_datasets.map(
        lambda x: {
            "tokens": ast.literal_eval(x["tokens"]),
            "ner_tags": ast.literal_eval(x["ner_tags"]),
        }
    )

    label_list = get_label_list(raw_datasets["train"])  # Assuming 'train' split exists and contains the labels

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))

    # Create label_to_id mapping
    label_to_id = {label: i for i, label in enumerate(label_list)}

    # Tokenization and alignment of labels
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id), batched=True
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=lambda p: compute_metrics(p, label_list),
    )

    # Train
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(args.path_to_save_model)
    tokenizer.save_pretrained(args.path_to_save_tokenizer)

    # Save the label list
    with open(args.path_to_save_label_list, "w") as f:
        json.dump(label_list, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--path_to_save_model", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--path_to_save_tokenizer", type=str, required=True, help="Path to save the tokenizer.")
    parser.add_argument("--path_to_save_label_list", type=str, required=True, help="Path to save the label list.")
    
    args = parser.parse_args()
    main(args)
