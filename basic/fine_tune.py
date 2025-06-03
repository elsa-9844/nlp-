import os
import warnings
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["WANDB_DISABLED"] = "true"

# Model dictionary
MODELS = {
    "facebook/bart-large-cnn": "facebook/bart-large-cnn",
    "t5-small": "t5-small",
    "allenai/led-base-16384": "allenai/led-base-16384",
    "sshleifer/distilbart-cnn-12-6": "sshleifer/distilbart-cnn-12-6",
    "google/pegasus-cnn_dailymail": "google/pegasus-cnn_dailymail"
}

# Dataset dictionary
DATASETS = {
    "xsum": ("xsum", None, 'document', 'summary'),
    "cnn_dailymail": ("cnn_dailymail", "3.0.0", 'article', 'highlights'),
    "pubmed": ("pubmed", None, 'text', 'abstract'),
    "samsum": ("samsum", None, 'dialogue', 'summary')
}

def convert_examples_to_features(batch, tokenizer, input_column, target_column):
    inputs = tokenizer(batch[input_column], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(batch[target_column], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

def fine_tune_model(model_name, dataset_option, dataset_name=None, custom_data=None):
    """Fine-tune the selected model with either an existing or custom dataset."""

    # Load model and tokenizer
    if model_name not in MODELS:
        print("Invalid model selected.")
        return
    model_ckpt = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

    # Load dataset
    if dataset_option == "existing":
        if dataset_name not in DATASETS:
            print("Invalid dataset selected.")
            return
        dataset_info = DATASETS[dataset_name]
        
        # **Fix: Add trust_remote_code=True**
        dataset = load_dataset(dataset_info[0], dataset_info[1], trust_remote_code=True) if dataset_info[1] else load_dataset(dataset_info[0], trust_remote_code=True)
        
        dataset = dataset.map(lambda batch: convert_examples_to_features(batch, tokenizer, dataset_info[2], dataset_info[3]), batched=True)

    elif dataset_option == "custom":
        if custom_data is None:
            print("No custom dataset provided.")
            return
        try:
            # Load the provided custom CSV
            custom_data = pd.read_csv(r"C:\Users\dell8\OneDrive\Documents\onedrive\Desktop\folder\add.csv")
            print("Custom dataset loaded successfully.")
        except Exception as e:
            print(f"Error reading custom CSV file: {e}")
            return

        try:
            # Load another dataset (smb.csv)
            df = pd.read_csv(r"C:\Users\dell8\OneDrive\Documents\onedrive\Desktop\folder\smb.csv")
            print("smb.csv loaded successfully.")
        except Exception as e:
            print(f"Error reading smb.csv file: {e}")
            return
        custom_data = pd.read_csv(r"C:\Users\dell8\OneDrive\Documents\onedrive\Desktop\folder\add.csv")
        # Combine datasets
        df_combined = pd.concat([custom_data, df], ignore_index=True)
        
        # Ensure 'id' column exists before type conversion
        if 'id' in df_combined.columns:
            df_combined['id'] = df_combined['id'].astype(str)

        print("Combined dataset preview:\n", df_combined.head())

        # Check required columns
        if 'dialogue' not in df_combined.columns or 'summary' not in df_combined.columns:
            print("Missing required columns ('dialogue' and 'summary'). Exiting.")
            return

        # Split data into train and validation sets
        train_df, val_df = train_test_split(df_combined, test_size=0.1, random_state=42)

        # Convert to Hugging Face dataset format
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        # Tokenize datasets
        train_dataset = train_dataset.map(lambda batch: convert_examples_to_features(batch, tokenizer, 'dialogue', 'summary'), batched=True)
        val_dataset = val_dataset.map(lambda batch: convert_examples_to_features(batch, tokenizer, 'dialogue', 'summary'), batched=True)

    else:
        print("Invalid dataset option.")
        return


    # Set training arguments
    trainer_args = TrainingArguments(
        output_dir='model',
        num_train_epochs=1,
        warmup_steps=500,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=500,
        logging_dir='./logs',
        gradient_accumulation_steps=16,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=trainer_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        train_dataset=dataset["train"] if dataset_option == "existing" else train_dataset,
        eval_dataset=dataset["validation"] if dataset_option == "existing" else val_dataset
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save model
    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")
    print("Model and tokenizer saved!")
