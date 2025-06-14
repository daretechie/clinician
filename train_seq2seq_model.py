import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import sentencepiece
import evaluate
import nltk
import numpy as np

# Attempt to download 'punkt' for nltk.sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    try:
        nltk.download('punkt', quiet=True) # quiet=True to reduce verbosity during download
        print("'punkt' downloaded successfully.")
    except Exception as e:
        print(f"Error downloading 'punkt': {e}. ROUGE scoring might be affected if sentence tokenization fails.")


def main():
    print("Starting Seq2Seq model training process with ROUGE metrics...")

    # --- Configuration ---
    model_checkpoint = "t5-small"
    input_column = "input_text"
    target_column = "Clinician"
    max_input_length = 1024
    max_target_length = 512
    prefix = "generate clinician response: "

    # --- 1. Load Data ---
    print("Loading data from data/train.csv...")
    try:
        df = pd.read_csv("data/train.csv")
        print(f"Data loaded. Initial shape: {df.shape}")
    except FileNotFoundError:
        print("Error: data/train.csv not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Prepare Text Columns & Handle Missing Values ---
    print("Preparing input and target columns...")
    df = df[['Prompt', target_column]].copy()
    df.rename(columns={'Prompt': input_column}, inplace=True)

    df.dropna(subset=[input_column, target_column], inplace=True)
    print(f"Dropped NaNs. Shape after cleaning: {df.shape}")

    if df.empty:
        print("Error: DataFrame is empty after dropping NaNs. Cannot proceed.")
        return

    # --- 3. Convert to Hugging Face Dataset ---
    print("Converting pandas DataFrame to Hugging Face Dataset...")
    raw_dataset = Dataset.from_pandas(df)
    print(f"Dataset created. Features: {raw_dataset.features}")

    # --- 4. Split Dataset ---
    print("Splitting dataset into training (80%) and validation (20%)...")
    if len(raw_dataset) < 2 :
        print("Error: Not enough data to split into training and validation sets.")
        if len(raw_dataset) == 1:
            print("Using the single sample for both training and validation for a structural run.")
            tokenized_datasets = DatasetDict({
                'train': raw_dataset,
                'validation': raw_dataset
            })
        else:
            return
    else:
        split_datasets = raw_dataset.train_test_split(test_size=0.2, seed=42)
        tokenized_datasets = DatasetDict({
            'train': split_datasets['train'],
            'validation': split_datasets['test']
        })

    print(f"Dataset split. Training size: {len(tokenized_datasets['train'])}, Validation size: {len(tokenized_datasets['validation'])}")

    # --- 5. Load Tokenizer and Model ---
    print(f"Loading tokenizer and model for '{model_checkpoint}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        print("Tokenizer and model loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("This might be due to a missing backend (PyTorch, TensorFlow, or Flax). Ensure one is installed.")
        return

    # --- 6. Preprocessing Function ---
    print("Defining preprocessing function...")
    def preprocess_function(examples):
        inputs = [prefix + str(doc) for doc in examples[input_column]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        targets_text = [str(text) for text in examples[target_column]]
        labels = tokenizer(text_target=targets_text, max_length=max_target_length, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # --- 7. Apply Preprocessing ---
    print("Applying preprocessing function to datasets...")
    try:
        tokenized_datasets = tokenized_datasets.map(preprocess_function, batched=True)
        print("Preprocessing applied.")
        print(f"Columns in tokenized training set: {tokenized_datasets['train'].column_names}")
    except Exception as e:
        print(f"Error during dataset mapping/tokenization: {e}")
        import traceback
        traceback.print_exc()
        return

    columns_to_remove_explicit = [input_column, target_column]
    if '__index_level_0__' in tokenized_datasets['train'].column_names:
        columns_to_remove_explicit.append('__index_level_0__')

    final_cols_to_remove = [c for c in columns_to_remove_explicit if c in tokenized_datasets['train'].column_names]
    if final_cols_to_remove:
        print(f"Removing columns: {final_cols_to_remove}")
        tokenized_datasets = tokenized_datasets.remove_columns(final_cols_to_remove)

    # --- 8. Define compute_metrics function for ROUGE scores ---
    print("Defining compute_metrics function for ROUGE scores...")
    rouge_scorer = evaluate.load('rouge')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # predictions will be token IDs if predict_with_generate=True
        # labels will be token IDs

        # Decode generated summaries (predictions)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = rouge_scorer.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Extract specific ROUGE scores
        result = {key: value * 100 for key, value in result.items()} # Changed to show percentage scores

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    print("compute_metrics function defined.")

    # --- 9. Define Training Arguments ---
    print("Defining training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results_seq2seq',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=2, # Reduced batch size further for memory
        per_device_eval_batch_size=2,  # Reduced batch size further for memory
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1, # Reduced epochs for quicker test, was 3
        predict_with_generate=True,
        fp16=False,
        # report_to="none", # Explicitly disable other reporting like wandb if not set up
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # --- 10. Initialize Trainer ---
    print("Initializing Seq2SeqTrainer with compute_metrics...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator, # Add data collator
        compute_metrics=compute_metrics # Add compute_metrics
    )
    print("Seq2SeqTrainer initialized.")

    # --- 11. Train Model ---
    print("Starting model training...")
    try:
        trainer.train()
        print("Model training completed.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        # Do not return here if we want to attempt saving even if training fails partially
        # However, if backend isn't found, model won't even load, so training won't start.

    # --- 12. Save Model ---
    print("Saving the fine-tuned model (if training started)...")
    try:
        # Check if model was loaded, otherwise trainer.model might not exist or be valid
        if 'model' in locals() and model is not None:
             trainer.save_model("./fine_tuned_t5_model_with_rouge")
             print("Model saved to ./fine_tuned_t5_model_with_rouge")
        else:
            print("Model was not loaded or trained, skipping save.")
    except Exception as e:
        print(f"Error saving model: {e}")
        # If trainer didn't initialize or run, saving might fail.
        # This is a fallback.
        if 'model' in locals() and hasattr(model, 'save_pretrained'):
            try:
                model.save_pretrained("./fine_tuned_t5_model_with_rouge_fallback")
                tokenizer.save_pretrained("./fine_tuned_t5_model_with_rouge_fallback")
                print("Fallback: Model saved using model.save_pretrained().")
            except Exception as e_fallback:
                 print(f"Fallback model save also failed: {e_fallback}")


    print("train_seq2seq_model.py script finished.")

if __name__ == '__main__':
    main()
