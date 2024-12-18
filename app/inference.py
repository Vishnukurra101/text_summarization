from transformers import BartTokenizer
import torch


def generate_summary(model, tokenizer, article, device):
    """
    Generates a summary for the input article using the fine-tuned BART model.
    """
    inputs = tokenizer(article, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device

    with torch.no_grad():  # Avoid gradient calculation during inference
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
