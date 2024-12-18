from transformers import BartForConditionalGeneration
import torch


def load_model():
    """
    Loads the pre-trained or fine-tuned BART model from the 'Model' directory.
    """
    model = BartForConditionalGeneration.from_pretrained('Model')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, device


def save_model(model, model_path):
    """
    Saves the fine-tuned model and tokenizer to the specified path.
    """
    model.save_pretrained(model_path)
    return model_path
