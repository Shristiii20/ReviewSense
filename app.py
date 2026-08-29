import streamlit as st
import torch
import shap
import numpy as np
import streamlit.components.v1 as components

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ReviewSense",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2.5rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .positive {
        background-color: rgba(34, 197, 94, 0.12);
        border-color: rgba(34, 197, 94, 0.3);
    }

    .negative {
        background-color: rgba(239, 68, 68, 0.12);
        border-color: rgba(239, 68, 68, 0.3);
    }

    .sentiment {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .confidence {
        font-size: 1.1rem;
        color: #d1d5db;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 3rem;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_DIR = Path(__file__).parent / "models" / "distilbert"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.eval()

    return tokenizer, model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    tokenizer, model = load_model()

except Exception as e:

    st.error(
        "⚠️ Could not load the trained DistilBERT model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def load_explainer():

    def predict_proba(texts):

        inputs = tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

        return probabilities.numpy()


    masker = shap.maskers.Text(tokenizer)

    explainer = shap.Explainer(
        predict_proba,
        masker
    )

    return explainer


# ============================================================
# LOAD SHAP
# ============================================================

try:

    explainer = load_explainer()

except Exception as e:

    explainer = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🧠 ReviewSense</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Transformer-Based Product Review Intelligence'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Analyze a Product Review")

review = st.text_area(
    "Enter your review below:",
    height=150,
    placeholder=(
        "Example: The battery life is amazing, "
        "but the camera quality is disappointing."
    )
)


# ============================================================
# ANALYSIS BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze Review",
    use_container_width=True
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[0][prediction].item()

    return prediction, confidence, probabilities


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if not review.strip():

        st.warning(
            "Please enter a product review first."
        )

    else:

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction, confidence, probabilities = predict_sentiment(
            review
        )


        # ----------------------------------------------------
        # LABEL MAPPING
        # ----------------------------------------------------

        if prediction == 0:

            sentiment = "Negative"
            emoji = "🔴"
            css_class = "negative"

        else:

            sentiment = "Positive"
            emoji = "🟢"
            css_class = "positive"


        # ----------------------------------------------------
        # RESULT CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <div class="sentiment">
                    {emoji} {sentiment}
                </div>

                <div class="confidence">
                    Confidence: {confidence * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # PROBABILITY BREAKDOWN
        # ----------------------------------------------------

        st.markdown("### Prediction Probabilities")

        negative_probability = probabilities[0][0].item()
        positive_probability = probabilities[0][1].item()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Negative",
                f"{negative_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Positive",
                f"{positive_probability * 100:.2f}%"
            )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.progress(
            positive_probability,
            text=(
                f"Positive probability: "
                f"{positive_probability * 100:.2f}%"
            )
        )


        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.write("**Model**")
            st.write("DistilBERT")

        with col2:

            st.write("**Task**")
            st.write("Sentiment Classification")

        with col3:

            st.write("**Classes**")
            st.write("Positive / Negative")


        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        if explainer is not None:

            st.divider()

            st.markdown(
                "### 🧠 Why did the model predict this?"
            )

            st.caption(
                "Words highlighted toward Positive or Negative "
                "show their contribution to the model's prediction."
            )

            with st.spinner(
                "Generating explanation..."
            ):

                try:

                    shap_values = explainer([review])

                    shap_html = shap.plots.text(
                        shap_values[0],
                        display=False
                    )

                    components.html(
                        shap.getjs() + shap_html,
                        height=450,
                        scrolling=True
                    )

                except Exception as e:

                    st.warning(
                        "SHAP explanation could not be generated."
                    )

                    st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ReviewSense • NLP + Transformers + Explainable ML
    </div>
    """,
    unsafe_allow_html=True
)