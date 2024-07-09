import argparse
import ast
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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

def get_predictions(tokenizer, model, dataset, label_list):
    # Create label_to_id mapping
    label_to_id = {label: i for i, label in enumerate(label_list)}

    # Tokenize and align the new dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id), batched=True
    )

    # Create a Trainer instance
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
    )

    # Get predictions
    predictions, labels, metrics = trainer.predict(tokenized_dataset)

    # Process predictions
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) and convert indices to label names
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_predictions

def main(args):
    # Load the saved model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(args.path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(args.path_to_tokenizer)

    # Load label list
    with open(args.path_to_label_list, "r") as f:
        label_list = json.load(f)

    # Load the new test set
    new_test_dataset = load_dataset(args.dataset_name, split="test")

    # Preprocess the new test dataset
    new_test_dataset = new_test_dataset.map(
        lambda x: {
            "tokens": ast.literal_eval(x["tokens"]),
            "ner_tags": ast.literal_eval(x["ner_tags"]),
        }
    )

    # Get predictions
    predictions = get_predictions(tokenizer, model, new_test_dataset, label_list)

    # Save the predictions to a .tsv file
    with open(args.output_file, "w") as f:
        for tokens, preds in zip(new_test_dataset["tokens"], predictions):
            for token, pred in zip(tokens, preds):
                f.write(f"{token}\t{pred}\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for NER model.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the new dataset.")
    parser.add_argument("--path_to_model", type=str, required=True, help="Path to the saved model.")
    parser.add_argument("--path_to_tokenizer", type=str, required=True, help="Path to the saved tokenizer.")
    parser.add_argument("--path_to_label_list", type=str, required=True, help="Path to the saved label list.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the predictions in .tsv format.")

    args = parser.parse_args()
    main(args)
