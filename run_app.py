from flask import Flask, request, render_template, jsonify
from app.model import load_model
from app.inference import generate_summary
from transformers import BartTokenizer


app = Flask(__name__)

# Load model and tokenizer from the 'Model' directory
model, device = load_model()
tokenizer = BartTokenizer.from_pretrained('Model')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the article from the form
    article = request.form['article']

    if not article:
        return render_template('index.html', error="No article provided")

    # Generate summary
    summary = generate_summary(model, tokenizer, article, device)

    # Return the result with the summary
    return render_template('index.html', article=article, summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
