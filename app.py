import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL & DATA ----------------
@st.cache_resource
def load_models():
    vectorizer = joblib.load("TF-IDF.pkl")
    model = joblib.load("LinearSVC.pkl")   # LinearSVC model
    return vectorizer, model

@st.cache_data
def load_data():
    return pd.read_csv("final_sample_dataset.csv")

vectorizer, model = load_models()
df = load_data()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
}
.subtitle {
    font-size: 18px;
    color: #6b7280;
}
.card {
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.positive {
    background-color: #dcfce7;
    color: #166534;
}
.neutral {
    background-color: #e0e7ff;
    color: #3730a3;
}
.negative {
    background-color: #fee2e2;
    color: #991b1b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🧠 Sentiment Analysis System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Positive · Neutral · Negative Text Classification</div>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    text_column = st.selectbox(
        "Select Text Column",
        df.columns
    )

    st.markdown("---")
    st.write("📊 Total Rows:", df.shape[0])
    st.write("📁 Total Columns:", df.shape[1])
    st.markdown("👨‍💻 Developed by **Sahil**")

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([1.3, 1])

# ---------------- DATA PREVIEW ----------------
with col1:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- SINGLE TEXT PREDICTION ----------------
with col2:
    st.subheader("✍️ Analyze Text")

    user_text = st.text_area(
        "Enter text for sentiment analysis",
        height=180,
        placeholder="Type or paste your text here..."
    )

    st.caption(f"🧾 Characters: {len(user_text)}")

    if st.button("🔍 Predict Sentiment", use_container_width=True):
        if user_text.strip() == "":
            st.warning("⚠️ Please enter some text")
        else:
            vector = vectorizer.transform([user_text])
            prediction = model.predict(vector)[0]  # STRING LABEL

            if prediction.lower() == "positive":
                css_class = "positive"
                sentiment = "Positive 😊"
            elif prediction.lower() == "neutral":
                css_class = "neutral"
                sentiment = "Neutral 😐"
            else:
                css_class = "negative"
                sentiment = "Negative 😠"

            st.markdown("### 📊 Prediction Result")
            st.markdown(
                f"<div class='card {css_class}'>{sentiment}</div>",
                unsafe_allow_html=True
            )

# ---------------- BULK PREDICTION ----------------
st.markdown("---")
st.subheader("📦 Bulk Sentiment Prediction")

if st.button("🚀 Analyze Entire Dataset"):
    text_data = df[text_column].astype(str)

    vectors = vectorizer.transform(text_data)
    predictions = model.predict(vectors)  # STRING LABELS

    result_df = df.copy()
    result_df["Sentiment"] = predictions

    st.success("✅ Sentiment analysis completed")
    st.dataframe(result_df.head(20), use_container_width=True)