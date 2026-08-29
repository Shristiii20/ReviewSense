# ReviewSense

ReviewSense is a transformer-based sentiment analysis system for product reviews.

The project fine-tunes DistilBERT to classify reviews as **Positive** or **Negative** and uses SHAP to show which words influenced the model's prediction.

## What it does

- Classifies product reviews into Positive or Negative
- Uses a fine-tuned DistilBERT model
- Shows prediction probabilities and confidence
- Provides SHAP-based explanations for predictions
- Includes a Streamlit interface for testing reviews
- Compares the transformer model with a TF-IDF baseline

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Scikit-learn
- SHAP
- Streamlit
- Pandas
- NumPy

## Project Structure

```text
ReviewSense/
│
├── data/
│   └── Dataset files
│
├── models/
│   └── Trained model files
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_tfidf_baseline.ipynb
│   ├── 03_distilbert.ipynb
│   └── 04_error_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── baseline.py
│   ├── model.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md