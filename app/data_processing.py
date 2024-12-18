from datasets import load_dataset
from transformers import BartTokenizer


def load_and_process_data():
    """
    Loads the CNN/DailyMail dataset and tokenizes the articles and summaries.
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    tokenizer = BartTokenizer.from_pretrained('lucadiliello/bart-small')

    def preprocess_function(examples):
        # Tokenize the articles and summaries (highlights)
        inputs = tokenizer(examples['article'], padding='max_length', truncation=True, max_length=512)
        labels = tokenizer(examples['highlights'], padding='max_length', truncation=True, max_length=150)

        # Ensure that labels are correctly aligned with the input format
        inputs['labels'] = labels['input_ids']
        return inputs

    # Apply the preprocessing function to the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    return tokenized_datasets
