from transformers import Trainer, TrainingArguments
from app.data_processing import load_and_process_data
from app.model import load_model, save_model


def train_model():
    # Load the dataset
    tokenized_datasets = load_and_process_data()

    # Load the model and tokenizer (from pre-trained model or fine-tuned model)
    model, device = load_model()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # Save checkpoints in 'results' folder
        num_train_epochs=3,             # Number of epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,   # Batch size for evaluation
        warmup_steps=500,               # Warmup steps
        weight_decay=0.01,              # Weight decay for regularization
        logging_dir='./logs',           # Directory for logs
        report_to="none",               # Disable reporting to any platform (optional)
        save_steps=500,                 # Save model every 500 steps
        save_total_limit=3,             # Keep a maximum of 3 checkpoints
        remove_unused_columns=False,    # Keep all columns
        load_best_model_at_end=True,    # Load the best model after training
        eval_strategy="steps",          # Evaluate every 'eval_steps' steps
        eval_steps=500,                 # Number of steps between evaluations
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model.to(device),          # Move model to the correct device (GPU/CPU)
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model and tokenizer to the 'Model' folder
    save_model(model, 'Model')


if __name__ == "__main__":
    train_model()
