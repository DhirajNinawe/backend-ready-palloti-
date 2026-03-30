"""
Health AI — Flask API Backend
Endpoints:
  POST /predict  — Upload a medical file (CSV/XLSX/PDF/PNG/JPG) and get predictions.
  POST /chat     — Ask a health question related to the predicted disease.
"""

import os
import re
import pickle
import traceback

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

# ── Optional OCR / PDF deps (graceful degradation) ──────────────────────────
try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ============================================================
#  CONFIG
# ============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "GROK_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ============================================================
#  FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app, origins="*")   # allow requests from the HTML frontend


# ============================================================
#  LOAD MODELS
# ============================================================
def load_model(name: str):
    path = os.path.join(BASE_DIR, f"{name}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


models = {}
for m in ["diabetes", "heart", "asthma"]:
    try:
        models[m] = load_model(m)
        print(f"[✓] Loaded model: {m}")
    except Exception as e:
        print(f"[✗] Could not load model '{m}': {e}")


# ============================================================
#  HELPERS
# ============================================================
def detect_csv_type(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    if "glucose" in cols:
        return "diabetes"
    if any(c in cols for c in ["thalach", "cp", "chol"]):
        return "heart"
    if any(c in cols for c in ["wheezing", "lungfunctionfev1", "shortnessofbreath"]):
        return "asthma"
    return "unknown"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df.select_dtypes(include=["number"])


def run_prediction(disease: str, df: pd.DataFrame):
    if disease not in models:
        raise ValueError(f"No model loaded for '{disease}'")

    model  = models[disease]
    df_num = preprocess(df)

    if df_num.empty:
        raise ValueError("No numeric columns found in the data")

    expected = model.n_features_in_

    if df_num.shape[1] > expected:
        df_num = df_num.iloc[:, :expected]
    elif df_num.shape[1] < expected:
        for i in range(expected - df_num.shape[1]):
            df_num[f"pad_{i}"] = 0

    # Use numpy array to bypass sklearn feature-name validation.
    # This is needed when columns come from image/PDF extraction (0,1,2…)
    # and don't match the training feature names (age, glucose, bmi…).
    X = df_num.values
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return pred, proba


def future_risk(prob: float) -> float:
    return min(prob + 0.10, 1.0)


def extract_text_from_file(file, ext: str) -> str:
    text = ""
    if ext in ("png", "jpg", "jpeg"):
        if not OCR_AVAILABLE:
            raise RuntimeError("pytesseract / Pillow not installed")
        img  = Image.open(file)
        text = pytesseract.image_to_string(img)
    elif ext == "pdf":
        if not PDF_AVAILABLE:
            raise RuntimeError("pdfplumber not installed")
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "")
    return text


def features_from_text(text: str) -> pd.DataFrame:
    nums = list(map(float, re.findall(r"\d+\.?\d*", text)))
    nums = (nums + [0] * 10)[:10]
    return pd.DataFrame([nums])


def get_ai_advice(disease: str, prob: float) -> str:
    prompt = (
        f"A patient has a {disease} risk probability of {prob*100:.1f}%.\n\n"
        "Please provide:\n"
        "- A brief plain-language explanation of what this means.\n"
        "- 3–4 practical preventive tips.\n"
        "- Recommended lifestyle changes.\n\n"
        "Keep the response concise, warm, and medically responsible."
    )
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI advice temporarily unavailable. ({e})"


def get_chat_response(question: str, disease: str, prob: float) -> str:
    system_prompt = (
        f"You are a knowledgeable but warm AI health assistant. "
        f"The user has a {disease} risk score of {prob*100:.1f}%. "
        "Answer their questions clearly, concisely, and responsibly. "
        "Always recommend consulting a qualified physician for definitive advice."
    )
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
            model="llama-3.1-8b-instant",
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Chatbot temporarily unavailable. ({e})"


# ============================================================
#  ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "models_loaded": list(models.keys())})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file    = request.files["file"]
    ext     = file.filename.rsplit(".", 1)[-1].lower()
    disease = None
    df      = None

    try:
        # ── CSV ──────────────────────────────────────────────
        if ext == "csv":
            df      = pd.read_csv(file)
            disease = detect_csv_type(df)

        # ── Excel ────────────────────────────────────────────
        elif ext == "xlsx":
            df      = pd.read_excel(file)
            disease = detect_csv_type(df)

        # ── Image / PDF ──────────────────────────────────────
        elif ext in ("png", "jpg", "jpeg", "pdf"):
            text    = extract_text_from_file(file, ext)
            disease = "diabetes"          # safe default for unstructured input
            df      = features_from_text(text)

        else:
            return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

        if disease == "unknown":
            return jsonify({"error": "Could not detect dataset type from column names"}), 422

        if disease not in models:
            return jsonify({"error": f"No model loaded for '{disease}'"}), 503

        prediction, probability = run_prediction(disease, df)
        future                  = future_risk(probability)
        advice                  = get_ai_advice(disease, probability)

        return jsonify({
            "disease":    disease,
            "prediction": prediction,
            "probability": round(probability, 4),
            "future":      round(future, 4),
            "advice":      advice,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    body     = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "").strip()
    disease  = body.get("disease", "").strip()
    prob     = float(body.get("prob", 0))

    if not question:
        return jsonify({"error": "question is required"}), 400
    if not disease:
        return jsonify({"error": "disease is required"}), 400

    response = get_chat_response(question, disease, prob)
    return jsonify({"response": response})


# ============================================================
#  ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    print("\n🚀 Health AI Flask API starting on http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000, host="127.0.0.1")
