import json
import shap
import numpy as np
import torch
import streamlit as st

from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Shristiii050/reviewsense-distilbert"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    # Download calibration parameters
    calibration_file = hf_hub_download(
        repo_id=MODEL_ID,
        filename="calibration.json"
    )

    with open(calibration_file, "r") as f:
        calibration = json.load(f)

    TEMPERATURE = calibration["temperature"]

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device, TEMPERATURE

# ============================================================
# INITIALIZE MODEL
# ============================================================

tokenizer, model, device, TEMPERATURE = load_model()
# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer():

    masker = shap.maskers.Text(tokenizer)

    def predict_proba(texts):

        inputs = tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            logits = outputs.logits

            calibrated_logits = (
                logits / TEMPERATURE
            )

            probabilities = torch.softmax(
                calibrated_logits,
                dim=-1
            )

        return probabilities.cpu().numpy()

    explainer = shap.Explainer(
        predict_proba,
        masker
    )

    return explainer


explainer = create_shap_explainer()
# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_review(text):

    shap_values = explainer(
        [text],
        max_evals=300
    )

    return shap_values
# ============================================================
# PREDICTION
# ============================================================

def predict_review(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        # Raw probabilities
        raw_probs = torch.softmax(
            logits,
            dim=-1
        )

        # Temperature-scaled probabilities
        calibrated_logits = (
            logits / TEMPERATURE
        )

        calibrated_probs = torch.softmax(
            calibrated_logits,
            dim=-1
        )

    predicted_class = torch.argmax(
        calibrated_probs,
        dim=-1
    ).item()

    labels = {
        0: "NEGATIVE",
        1: "POSITIVE"
    }

    sentiment = labels[predicted_class]

    confidence = calibrated_probs[
        0,
        predicted_class
    ].item()

    if confidence >= 0.95:
        confidence_level = "Very High"
    elif confidence >= 0.90:
        confidence_level = "High"
    elif confidence >= 0.75:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "negative_probability": calibrated_probs[0, 0].item(),
        "positive_probability": calibrated_probs[0, 1].item(),
        "raw_confidence": raw_probs[0, predicted_class].item()
    }


# ============================================================
# HEADER
# ============================================================

st.title("🔍 ReviewSense")

st.subheader(
    "AI-Powered Product Review Sentiment Analysis"
)

st.write(
    "Analyze a product review using a fine-tuned "
    "DistilBERT sentiment classification model."
)


# ============================================================
# MODEL INFO
# ============================================================

with st.expander("About ReviewSense"):

    st.write(
        """
        **ReviewSense** uses a fine-tuned DistilBERT model
        to classify product reviews as positive or negative.

        **Test Set Performance**

        - Accuracy: 93.18%
        - Precision: 93.61%
        - Recall: 93.06%
        - F1 Score: 93.33%

        The model also uses temperature scaling to improve
        confidence calibration.
        """
    )


# ============================================================
# REVIEW INPUT
# ============================================================

st.markdown("### 📝 Enter a product review")

review = st.text_area(
    "Review",
    placeholder=(
        "Example: The battery life is amazing, "
        "but the camera quality is disappointing."
    ),
    height=150,
    label_visibility="collapsed"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔮 Analyze Review",
    use_container_width=True
):

    if not review.strip():

        st.warning(
            "Please enter a review first."
        )

    else:

        result = predict_review(
            review.strip()
        )

        sentiment = result["sentiment"]
        confidence = result["confidence"]

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown("### 🎯 Prediction")

        if sentiment == "POSITIVE":

            st.success(
                f"## 🟢 POSITIVE\n\n"
                f"Confidence: **{confidence:.2%}**"
            )

        else:

            st.error(
                f"## 🔴 NEGATIVE\n\n"
                f"Confidence: **{confidence:.2%}**"
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.markdown("### 📊 Probability Breakdown")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Negative",
                f"{result['negative_probability']:.2%}"
            )

            st.progress(
                result["negative_probability"]
            )

        with col2:

            st.metric(
                "Positive",
                f"{result['positive_probability']:.2%}"
            )

            st.progress(
                result["positive_probability"]
            )


        # ----------------------------------------------------
        # CONFIDENCE LEVEL
        # ----------------------------------------------------

        st.markdown("### 🎯 Model Confidence")

        st.write(
            f"**{result['confidence_level']}**"
        )

        st.progress(
            confidence
        )


        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🔬 Technical Details"
        ):

            st.write(
                f"**Device:** {device}"
            )

            st.write(
                f"**Raw confidence:** "
                f"{result['raw_confidence']:.2%}"
            )

            st.write(
                f"**Calibrated confidence:** "
                f"{confidence:.2%}"
            )

            st.write(
                f"**Temperature:** "
                f"{TEMPERATURE}"
            )
# ----------------------------------------------------
# SHAP EXPLANATION
# ----------------------------------------------------

st.markdown("### 🔎 Why did the model predict this?")

with st.spinner("Generating explanation..."):

    shap_values = explain_review(
        review.strip()
    )

values = shap_values.values[0]
tokens = shap_values.data[0]
# Positive-class SHAP contributions
if values.ndim == 2:

    positive_values = values[:, 1]

else:

    positive_values = values


token_data = []

for token, value in zip(
    tokens,
    positive_values
):

    token = str(token).strip()

    if not token:
        continue

    token_data.append(
        {
            "token": token,
            "contribution": float(value)
        }
    )

if token_data:

    positive_tokens = sorted(
        token_data,
        key=lambda x: x["contribution"],
        reverse=True
    )[:5]

    negative_tokens = sorted(
        token_data,
        key=lambda x: x["contribution"]
    )[:5]


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("#### Positive contributors")

        for item in positive_tokens:

            st.write(
                f"**{item['token']}**  "
                f"`+{item['contribution']:.3f}`"
            )


    with col2:

        st.markdown("#### Negative contributors")

        for item in negative_tokens:

            st.write(
                f"**{item['token']}**  "
                f"`{item['contribution']:.3f}`"
            )

# ============================================================
# EXAMPLE REVIEWS
# ============================================================

st.markdown("---")

st.markdown(
    "### 💡 Try an example"
)

examples = [
    "This product is absolutely fantastic and works perfectly.",
    "The product is terrible, disappointing, and completely useless.",
    "I loved the battery life, but the phone overheats constantly.",
    "Not bad, but definitely not worth the price."
]

for example in examples:

    st.code(
        example,
        language=None
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ReviewSense • DistilBERT Sentiment Classification"
)
