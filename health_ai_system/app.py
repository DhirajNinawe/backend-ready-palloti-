import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from groq import Groq

# ================= CONFIG =================
GROQ_API_KEY = "GROK_API_KEY"  
client = Groq(api_key=GROQ_API_KEY)

# ================= LOAD MODELS =================
models = {
    "diabetes": pickle.load(open("diabetes_model.pkl", "rb")),
    "heart": pickle.load(open("heart_model.pkl", "rb")),
    "asthma": pickle.load(open("asthma_model.pkl", "rb"))
}

# ================= PAGE =================
st.set_page_config(page_title="Health AI", layout="wide")
st.title("🧠 AI Health Intelligence System")

# ================= DETECT DATASET =================
def detect_csv_type(df):
    cols = [c.lower() for c in df.columns]

    if "glucose" in cols:
        return "diabetes"
    elif "thalach" in cols or "cp" in cols:
        return "heart"
    elif "wheezing" in cols or "lungfunctionfev1" in cols:
        return "asthma"

    return "unknown"

# ================= CLEAN DATA =================
def preprocess(df):
    df.columns = [c.lower() for c in df.columns]
    return df.select_dtypes(include=['number'])

# ================= PREDICT =================
def predict(disease, df):

    model = models[disease]
    df = preprocess(df)

    expected = model.n_features_in_

    if df.shape[1] > expected:
        df = df.iloc[:, :expected]
    elif df.shape[1] < expected:
        for i in range(expected - df.shape[1]):
            df[f"x{i}"] = 0

    try:
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
    except:
        return None, 0

    return pred, prob

# ================= FUTURE =================
def future_risk(prob):
    return min(prob + 0.1, 1.0)

# ================= AI ADVICE =================
def ai_advice(disease, prob):

    prompt = f"""
    Patient has {disease} risk of {prob*100:.2f}%.

    Give:
    - Simple explanation
    - Preventive advice
    - Lifestyle tips

    Keep it short and clear.
    """

    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant"
        )

        return chat.choices[0].message.content

    except Exception as e:
        return f"⚠️ AI advice unavailable ({str(e)})"

# ================= CHATBOT =================
def chatbot_response(question, disease, prob):

    prompt = f"""
    Patient has {disease} risk of {prob*100:.2f}%.

    Answer clearly:

    {question}
    """

    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant"
        )

        return chat.choices[0].message.content

    except Exception as e:
        return f"⚠️ Chatbot unavailable ({str(e)})"

# ================= UI =================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    # detect disease
    disease = detect_csv_type(df)

    if disease == "unknown":
        st.error("❌ Could not detect dataset type")
        st.stop()

    st.success(f"Detected: {disease.upper()}")

    # predict
    pred, prob = predict(disease, df)

    if pred is None:
        st.error("❌ Prediction failed (data mismatch)")
        st.stop()

    # ================= DIAGNOSIS =================
    st.subheader("🧠 Diagnosis")

    if pred == 1:
        st.error(f"{disease.upper()} Risk Detected ({prob*100:.2f}%)")
    else:
        st.success("✅ No major risk detected")

    # ================= FUTURE =================
    st.subheader("📈 Future Risk")

    future = future_risk(prob)
    st.write(f"{disease}: {future*100:.2f}%")

    # ================= GRAPH =================
    st.subheader("📊 Risk Graph")

    fig, ax = plt.subplots()
    ax.bar([disease, "Future"], [prob, future])
    ax.set_ylabel("Risk")

    st.pyplot(fig)

    # ================= AI ADVICE =================
    st.subheader("🤖 AI Clinical Advice")

    advice = ai_advice(disease, prob)
    st.info(advice)

    # ================= CHATBOT =================
    st.subheader("💬 AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask about your health...")

    if user_input:
        response = chatbot_response(user_input, disease, prob)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", response))

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")
