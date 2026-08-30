import json
import time

import shap
import numpy as np
import torch
import streamlit as st

from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Shristiii050/reviewsense-distilbert"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ReviewSense — AI Sentiment Analysis",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM UI / CSS
# ============================================================

st.markdown(
    """
    <style>
        /* ---------- GLOBAL ---------- */
        .stApp {
            background: #000000;
            color: #f5f7fa;
            font-family: Inter, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: #000000;
        }

        [data-testid="stHeader"] {
            background: #000000;
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* ---------- HEADER ---------- */
        .hero {
            text-align: center;
            padding: 0.8rem 0 1.5rem 0;
        }

        .hero-title {
            font-size: 3.05rem;
            line-height: 1.1;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0;
            background: linear-gradient(90deg, #ffffff 0%, #9fc4ff 55%, #5b8def 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            margin-top: 0.75rem;
            color: #d7dbe2;
            font-size: 1.15rem;
            font-weight: 500;
        }

        .hero-description {
            margin-top: 0.3rem;
            color: #7f8794;
            font-size: 0.95rem;
        }

        /* ---------- CARDS ---------- */
        .ui-card {
            background: #050505;
            border: 1px solid #24272d;
            border-radius: 12px;
            padding: 1.15rem;
            box-shadow: none;
        }

        .section-label {
            color: #f4f5f7;
            font-size: 1rem;
            font-weight: 650;
            margin-bottom: 0.7rem;
        }

        /* ---------- ABOUT ---------- */
        .about-copy {
            color: #aeb5c0;
            line-height: 1.7;
            font-size: 0.94rem;
        }

        .about-copy strong {
            color: #f1f3f6;
        }

        /* ---------- INPUT ---------- */
        .input-card {
            background: #050505;
            border: 1px solid #24272d;
            border-radius: 12px;
            padding: 1rem 1.1rem 1.1rem 1.1rem;
            margin-top: 1rem;
        }

        .stTextArea textarea {
            background: #020202 !important;
            color: #f5f7fa !important;
            border: 1px solid #30343b !important;
            border-radius: 9px !important;
            font-size: 0.98rem !important;
            line-height: 1.55 !important;
            padding: 0.9rem !important;
        }

        .stTextArea textarea:focus {
            border-color: #4f8cff !important;
            box-shadow: 0 0 0 1px #4f8cff !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 42px;
            border-radius: 8px;
            border: 1px solid #376bc6;
            background: #0b1830;
            color: #f3f6fb;
            font-weight: 650;
            transition: 0.18s ease;
        }

        .stButton > button:hover {
            border-color: #5c9aff;
            background: #10254a;
            color: #ffffff;
        }

        /* ---------- RESULT GRID ---------- */
        .prediction-card {
            min-height: 164px;
            background: #050505;
            border: 1px solid #24272d;
            border-left: 4px solid #63d471;
            border-radius: 12px;
            padding: 1.35rem 1.4rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .prediction-negative {
            border-left-color: #ff6464;
        }

        .prediction-label {
            font-size: 1.85rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            margin-bottom: 0.65rem;
        }

        .positive-text {
            color: #79d887;
        }

        .negative-text {
            color: #ff7777;
        }

        .status-dot {
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 0.6rem;
            vertical-align: 2px;
        }

        .dot-positive {
            background: #75d783;
            box-shadow: 0 0 0 3px rgba(117, 215, 131, 0.12);
        }

        .dot-negative {
            background: #ff6b6b;
            box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.12);
        }

        .confidence-label {
            color: #8f97a3;
            font-size: 0.88rem;
            margin-bottom: 0.2rem;
        }

        .confidence-value {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 750;
            letter-spacing: -0.03em;
        }

        .confidence-badge {
            display: inline-block;
            margin-left: 0.55rem;
            padding: 0.25rem 0.55rem;
            border-radius: 5px;
            background: #102416;
            border: 1px solid #24482c;
            color: #8ddd98;
            font-size: 0.75rem;
            font-weight: 650;
            vertical-align: 0.35rem;
        }

        .probability-card {
            background: #050505;
            border: 1px solid #24272d;
            border-radius: 12px;
            overflow: hidden;
        }

        .probability-row {
            padding: 1rem 1.15rem 0.9rem 1.15rem;
        }

        .probability-row + .probability-row {
            border-top: 1px solid #202329;
        }

        .prob-label {
            color: #b5bbc4;
            font-size: 0.84rem;
            margin-bottom: 0.18rem;
        }

        .prob-value {
            color: #f5f7fa;
            font-size: 1.65rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .progress-track {
            width: 100%;
            height: 7px;
            background: #20242b;
            border-radius: 999px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: #4388e8;
            border-radius: 999px;
        }

        /* ---------- LOWER GRID ---------- */
        .panel {
            background: #050505;
            border: 1px solid #24272d;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }

        .panel-title {
            color: #f4f5f7;
            font-size: 1.08rem;
            font-weight: 680;
            margin-bottom: 0.9rem;
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.68rem 0;
            border-bottom: 1px solid #1c1f24;
            font-size: 0.87rem;
        }

        .detail-row:last-child {
            border-bottom: none;
        }

        .detail-key {
            color: #8f97a3;
        }

        .detail-value {
            color: #dfe3e8;
            text-align: right;
        }

        .contributor-heading {
            font-size: 0.85rem;
            font-weight: 650;
            margin-bottom: 0.55rem;
        }

        .contributor-positive {
            color: #7edb8a;
        }

        .contributor-negative {
            color: #ff7777;
        }

        .contributor-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #090a0c;
            border: 1px solid #20242a;
            border-radius: 7px;
            padding: 0.48rem 0.65rem;
            margin-bottom: 0.45rem;
            font-size: 0.82rem;
        }

        .token {
            color: #e7eaf0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 68%;
        }

        .value-positive {
            color: #7edb8a;
            font-variant-numeric: tabular-nums;
        }

        .value-negative {
            color: #ff7777;
            font-variant-numeric: tabular-nums;
        }

        .shap-note {
            margin-top: 0.8rem;
            color: #737b87;
            font-size: 0.76rem;
        }

        /* ---------- EXAMPLES ---------- */
        .examples-title {
            color: #f4f5f7;
            font-size: 1.15rem;
            font-weight: 680;
            margin: 1.55rem 0 0.8rem 0;
        }

        .example-button > button {
            min-height: 78px;
            text-align: left;
            white-space: normal;
            background: #050505;
            border: 1px solid #24272d;
            color: #dfe3e8;
            font-weight: 550;
            line-height: 1.35;
        }

        .example-button > button:hover {
            background: #090b0e;
            border-color: #3e73bc;
        }

        /* ---------- EXPANDER ---------- */
        [data-testid="stExpander"] {
            border: 1px solid #24272d !important;
            border-radius: 10px !important;
            background: #050505 !important;
        }

        [data-testid="stExpander"] details summary {
            color: #e9edf2 !important;
        }

        /* ---------- DIVIDER / FOOTER ---------- */
        .soft-divider {
            height: 1px;
            background: #202329;
            margin: 1.4rem 0;
        }

        .footer {
            text-align: center;
            color: #646c78;
            font-size: 0.78rem;
            padding: 0.6rem 0 0.2rem 0;
        }

        /* ---------- MOBILE ---------- */
        @media (max-width: 800px) {
            .hero-title {
                font-size: 2.35rem;
            }

            .hero-subtitle {
                font-size: 1rem;
            }

            .confidence-value {
                font-size: 1.75rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔍 ReviewSense</div>
        <div class="hero-subtitle">
            AI-Powered Product Review Sentiment Analysis
        </div>
        <div class="hero-description">
            Enterprise-grade insights powered by DistilBERT
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    calibration_file = hf_hub_download(
        repo_id=MODEL_ID,
        filename="calibration.json"
    )

    with open(calibration_file, "r") as f:
        calibration = json.load(f)

    temperature = calibration["temperature"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device, temperature


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
            calibrated_logits = logits / TEMPERATURE

            probabilities = torch.softmax(
                calibrated_logits,
                dim=-1
            )

        return probabilities.cpu().numpy()

    return shap.Explainer(
        predict_proba,
        masker
    )


explainer = create_shap_explainer()


def explain_review(text):
    return explainer(
        [text],
        max_evals=300
    )


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

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        raw_probs = torch.softmax(
            logits,
            dim=-1
        )

        calibrated_logits = logits / TEMPERATURE

        calibrated_probs = torch.softmax(
            calibrated_logits,
            dim=-1
        )

    latency_ms = (time.perf_counter() - start_time) * 1000

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

    token_count = int(
        (inputs["attention_mask"][0] == 1).sum().item()
    )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "negative_probability": calibrated_probs[0, 0].item(),
        "positive_probability": calibrated_probs[0, 1].item(),
        "raw_confidence": raw_probs[0, predicted_class].item(),
        "token_count": token_count,
        "latency_ms": latency_ms,
    }


# ============================================================
# ABOUT
# ============================================================

with st.expander("ⓘ  About ReviewSense & Technology"):
    st.markdown(
        """
        <div class="about-copy">
            <strong>ReviewSense</strong> uses a fine-tuned DistilBERT
            transformer model to classify consumer reviews as positive
            or negative.
            <br><br>
            <strong>Test Set Performance</strong><br>
            Accuracy: 93.18% &nbsp;•&nbsp;
            Precision: 93.61% &nbsp;•&nbsp;
            Recall: 93.06% &nbsp;•&nbsp;
            F1 Score: 93.33%
            <br><br>
            The model also uses temperature scaling to improve
            confidence calibration and SHAP for token-level
            explainability.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# REVIEW INPUT
# ============================================================

if "review_input" not in st.session_state:
    st.session_state.review_input = ""


st.markdown(
    """
    <div class="input-card">
        <div class="section-label">Enter a product review</div>
    """,
    unsafe_allow_html=True,
)

review = st.text_area(
    "Review",
    key="review_input",
    placeholder=(
        "Example: The battery life is amazing, "
        "but the camera quality is disappointing."
    ),
    height=120,
    label_visibility="collapsed",
)

analyze_clicked = st.button(
    "✦  Analyze Sentiment",
    use_container_width=True
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_clicked:

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        review = review.strip()

        result = predict_review(review)

        sentiment = result["sentiment"]
        confidence = result["confidence"]

        # ----------------------------------------------------
        # TOP RESULT GRID
        # ----------------------------------------------------

        left, right = st.columns([1, 1.35], gap="medium")

        with left:
            if sentiment == "POSITIVE":
                prediction_html = f"""
                <div class="prediction-card">
                    <div class="prediction-label positive-text">
                        <span class="status-dot dot-positive"></span>
                        POSITIVE
                    </div>
                    <div class="confidence-label">Model Confidence</div>
                    <div>
                        <span class="confidence-value">
                            {confidence:.2%}
                        </span>
                        <span class="confidence-badge">
                            {result["confidence_level"]}
                        </span>
                    </div>
                </div>
                """
            else:
                prediction_html = f"""
                <div class="prediction-card prediction-negative">
                    <div class="prediction-label negative-text">
                        <span class="status-dot dot-negative"></span>
                        NEGATIVE
                    </div>
                    <div class="confidence-label">Model Confidence</div>
                    <div>
                        <span class="confidence-value">
                            {confidence:.2%}
                        </span>
                        <span class="confidence-badge">
                            {result["confidence_level"]}
                        </span>
                    </div>
                </div>
                """

            st.markdown(
                prediction_html,
                unsafe_allow_html=True
            )

        with right:
            negative_pct = result["negative_probability"] * 100
            positive_pct = result["positive_probability"] * 100
            st.html(
                f"""
                <div class="probability-card">
                    <div class="probability-row">
                        <div class="prob-label">
                            Negative Probability
                        </div>
                        <div class="prob-value">
                            {negative_pct:.2f}%
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill"
                                style="width:{negative_pct:.2f}%;">
                            </div>
                        </div>
                    </div>

                    <div class="probability-row">
                        <div class="prob-label">
                            Positive Probability
                        </div>
                        <div class="prob-value">
                            {positive_pct:.2f}%
                        </div>

                    <div class="progress-track">
                        <div class="progress-fill"
                            style="width:{positive_pct:.2f}%;">
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        with st.spinner("Generating explanation..."):
            shap_values = explain_review(review)

        values = shap_values.values[0]
        tokens = shap_values.data[0]

        if values.ndim == 2:
            positive_values = values[:, 1]
        else:
            positive_values = values

        token_data = []

        for token, value in zip(tokens, positive_values):
            token = str(token).strip()

            if not token:
                continue

            token_data.append(
                {
                    "token": token,
                    "contribution": float(value)
                }
            )

        positive_tokens = sorted(
            [x for x in token_data if x["contribution"] > 0],
            key=lambda x: x["contribution"],
            reverse=True
        )[:5]
        negative_tokens = sorted(
            [x for x in token_data if x["contribution"] < 0],
            key=lambda x: x["contribution"]
        )[:5]

        # ----------------------------------------------------
        # LOWER PANELS
        # ----------------------------------------------------

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            with st.expander("⚙  Technical Details", expanded=True):
                st.markdown(
                    f"""
                    <div class="detail-row">
                        <span class="detail-key">Model</span>
                        <span class="detail-value">
                            DistilBERT (Fine-tuned)
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-key">Device</span>
                        <span class="detail-value">
                            {str(device).upper()}
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-key">Tokens</span>
                        <span class="detail-value">
                            {result["token_count"]}
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-key">Inference Latency</span>
                        <span class="detail-value">
                            {result["latency_ms"]:.0f} ms
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-key">Temperature</span>
                        <span class="detail-value">
                            {TEMPERATURE:.2f}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            positive_html = ""

            for item in positive_tokens:
                positive_html += f"""
                <div class="contributor-row">
                    <span class="token">{item["token"]}</span>
                    <span class="value-positive">
                        +{item["contribution"]:.3f}
                    </span>
                </div>
                """

            negative_html = ""

            for item in negative_tokens:
                negative_html += f"""
                <div class="contributor-row">
                    <span class="token">{item["token"]}</span>
                    <span class="value-negative">
                        {item["contribution"]:.3f}
                    </span>
                </div>
                """

            st.html(
                f"""
                <div class="panel">
                    <div class="panel-title">
                        Why did the model predict this?
                    </div>

                    <div style="display:grid;
                                grid-template-columns:1fr 1fr;
                                gap:1rem;">

                        <div>
                            <div class="contributor-heading
                                        contributor-positive">
                                Positive Contributors
                            </div>
                            {positive_html}
                        </div>

                        <div>
                            <div class="contributor-heading
                                        contributor-negative">
                                Negative Contributors
                            </div>
                            {negative_html}
                        </div>

                    </div>

                    <div class="shap-note">
                        ⓘ Showing top token-level attributions
                        (SHAP values)
                    </div>
                </div>
                """
            )
# ============================================================
# EXAMPLE REVIEWS
# ============================================================
st.markdown(
    '<div class="examples-title">Try an Example</div>',
    unsafe_allow_html=True
)
examples = [
    ("Fantastic & Works Perfectly",
     "This product is absolutely fantastic and works perfectly."),
    ("Great Battery, Terrible Camera",
     "The battery life is excellent, but the camera quality is really disappointing."),
    ("Completely Useless & Broken",
     "This product is completely useless and broke after just a few days."),
    ("Not Bad, Overpriced",
     "It is not bad but definitely overpriced for what you get."),
]
example_cols = st.columns(4, gap="small")
def set_example(text):
    st.session_state.review_input = text

for col, (title, text) in zip(example_cols, examples):
    with col:
        st.button(
            title,
            key=f"example_{title}",
            use_container_width=True,
            on_click=set_example,
            args=(text,),
        )
# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer">
        ReviewSense &nbsp;•&nbsp; Built with Streamlit
        &nbsp;•&nbsp; Powered by DistilBERT
    </div>
    """,
    unsafe_allow_html=True
)
