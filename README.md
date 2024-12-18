# Text Summarization with Fine-Tuned BART

This project fine-tunes the **BART** model for text summarization tasks using the CNN/DailyMail dataset. The project includes the capability to fine-tune the model, save checkpoints, and deploy a Flask-based web app to generate article summaries interactively.

---

## Project Structure

```
root/
|
|-- .git/                  # Git version control
|-- app/                   # Contains scripts for model fine-tuning
|    |-- __init__.py       # Package initializer
|    |-- main.py           # Script to fine-tune the BART model
|    |-- data_processing.py # Handles data preprocessing
|    |-- inference.py      # Script for model inference
|    |-- model.py          # Defines the BART model structure
|
|-- Model/                 # Directory to store the fine-tuned BART model(model.safetensors not uploaded due to its size)
|
|-- results/               # Saves fine-tuning checkpoints(empty due to its size)
|
|-- templates/             # Contains HTML template for Flask app
|    |-- index.html        # Main HTML file for the web interface
|
|-- README.md              # Project documentation (this file)
|-- requirements.txt       # Required Python libraries
|-- run_app.py             # Flask app to generate text summaries
```

---

## Features

- **Fine-Tune BART Model**: 
   - Use `app/main.py` to fine-tune the BART model on custom data or the CNN/DailyMail dataset.
   - Save the model and checkpoints during the training process.

- **Text Summarization Web App**:
   - Run `run_app.py` to launch a Flask-based web application.
   - Input an article, and the fine-tuned model generates a summary.

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd root
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Fine-Tune the Model** (Optional):
   - Run the following to fine-tune the BART model:
     ```bash
     python app/main.py
     ```
   - Checkpoints will be saved in the `results/` folder.
   - The final fine-tuned model will be stored in the `Model/` directory.

4. **Run the Flask Application**:
   - Start the Flask web server with:
     ```bash
     python run_app.py
     ```
   - Open the app in your browser at: `http://127.0.0.1:5000/`

5. **Generate Summaries**:
   - Use the provided web interface to input articles and generate summaries.

---

## Dependencies

The required libraries are listed in `requirements.txt`:
- `torch`
- `transformers`
- `datasets`
- `matplotlib`
- `seaborn`
- `flask`

Install them via:
```bash
pip install -r requirements.txt
```

---

## Demo
1. Launch the Flask app with `python run_app.py`.
2. Input a news article or any text in the web app.
3. Click submit to see the summarized output.

---

## Results
- **Checkpoints**: All fine-tuning checkpoints are saved in `results/`.
- **Fine-Tuned Model**: The trained model is saved in the `Model/` folder.

---

## Future Improvements
- Improve UI design for the Flask web app.

---

## Acknowledgements
- Hugging Face's `transformers` library.
- Flask for web application development.
- CNN/DailyMail dataset for text summarization.

---

## Author
*Kurra Vishnuvardhan*

Feel free to fork, contribute, or report issues!
