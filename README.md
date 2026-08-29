# ReviewSense

ReviewSense is a transformer-based sentiment analysis system for product reviews.

It fine-tunes DistilBERT to classify reviews as **Positive** or **Negative**, and includes model evaluation, error analysis, and SHAP-based explanations to understand why the model makes a particular prediction.

## Overview

The project takes a product review as input and predicts its sentiment along with the probability of each class.

For example:

> "The battery life is amazing but the camera quality is disappointing."

The model identifies the overall sentiment while also providing an explanation of which words contributed to the prediction.

The project includes:

- DistilBERT fine-tuning for binary sentiment classification
- Evaluation using Accuracy, Precision, Recall and F1
- Confusion matrix analysis
- Error analysis based on review length and negation
- High-confidence error analysis
- SHAP explanations for individual predictions
- Streamlit interface for interactive predictions

## Model

**Base model:** DistilBERT

**Task:** Binary sentiment classification

**Classes:**

- `0` → Negative
- `1` → Positive

The model was fine-tuned on a product review sentiment dataset.

### Training Configuration

| Parameter | Value |
|---|---:|
| Model | DistilBERT |
| Epochs | 1 |
| Training steps | 2500 |
| Max sequence length | 128 |
| Task | Binary classification |

## Results

The fine-tuned model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 93.18% |
| Precision | 93.61% |
| Recall | 93.06% |
| F1 Score | 93.33% |

### Confusion Matrix

The test set produced:

|  | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 2272 | 163 |
| Actual Positive | 178 | 2387 |

This gives **341 incorrect predictions** out of the evaluation set.

## Error Analysis

I also looked at where the model performs well and where it tends to make mistakes.

### Performance by Review Length

| Review Length | Samples | Accuracy |
|---|---:|---:|
| Short | 605 | 93.39% |
| Medium | 1421 | 94.02% |
| Long | 1448 | 93.72% |
| Very Long | 1526 | 91.81% |

Performance remains fairly stable across short, medium and long reviews, with a noticeable drop for very long reviews.

### Reviews Containing Negation

| Contains Negation | Samples | Accuracy |
|---|---:|---:|
| No | 2593 | 93.87% |
| Yes | 2407 | 92.44% |

Reviews containing negation were slightly harder for the model to classify correctly.

### High-Confidence Errors

Out of the 341 incorrect predictions, **225 were high-confidence errors**.

These examples are particularly useful for understanding the model's limitations because the model was confident even when its prediction was incorrect.

## Explainability

ReviewSense uses **SHAP** to examine individual predictions.

The explanation highlights words according to their contribution to the model output. This makes it possible to inspect whether the model is relying on meaningful sentiment-bearing words such as:

- `amazing`
- `excellent`
- `disappointing`
- `terrible`

For example, for a review containing both positive and negative language, the SHAP explanation can show which terms pushed the prediction toward one class.

This provides a way to inspect the model beyond its final classification and confidence score.

## Web Application

The project includes a Streamlit interface where users can enter a product review and receive:

- Predicted sentiment
- Prediction confidence
- Probability for each class
- SHAP-based explanation of the prediction

### Example

Input:

> "The product looks good but the battery life is disappointing."

Output:

**Negative — 98.80% confidence**

The application also displays the probability distribution between Positive and Negative classes and a SHAP explanation of the prediction.

## Project Structure

ReviewSense/
│
├── app.py
│
├── models/
│   └── distilbert/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       └── ...
│
├── notebooks/
│   └── ...
│
├── requirements.txt
│
└── README.md