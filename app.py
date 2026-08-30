import re
import torch
import joblib
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np

app = FastAPI()

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_MODEL_PATH = "./models/deberta_finetuned_v579"
tokenizer_text = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
model_text = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH).to(DEVICE)
model_text.eval()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   
    text = re.sub(r"(https?:\/\/|www\.)[^\s]+", "", text)
    text = re.sub(r"\S+@\S+", " email ", text)         
    text = re.sub(r"\d+", " number ", text)            
    text = re.sub(r"<.*?>", " ", text)                 
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text(text):
    text = clean_text(text)
    enc = tokenizer_text(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model_text(**enc)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()   
        label = "Spam" if pred == 1 else "Ham"
    return label, float(probs[0]), float(probs[1])

MAX_LEN = 200
clf_url = load_model("./model_full/url_full/my_lstm_url_model.h5")
vectorizer_url = joblib.load("./model_full/url_full/tokenizer_url.pkl")
TRUSTED_DOMAINS = ["intra.company-hr.com", "hr.company.local"]

def is_trusted_domain(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.lower() in TRUSTED_DOMAINS

def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    domain = parsed.netloc.lower()
    path = parsed.path if parsed.path else "/"
    return domain + path

def predict_url(url: str):
    if not url:
        return "No URL"
    if is_trusted_domain(url):
        return "Safe URL (trusted)"

    norm_url = normalize_url(url)
    seq = vectorizer_url.texts_to_sequences([norm_url])
    X = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    prob = clf_url.predict(X)[0][0]
    label = "Malicious URL" if prob >= 0.5 else "Safe URL"
    return f"{label} (conf={prob:.2f})"

EMO_MODEL_DIR = "./models_emotion45"
rf_emotion = joblib.load(f"{EMO_MODEL_DIR}/rf_emotion_model.pkl")
vectorizer_emotion = joblib.load(f"{EMO_MODEL_DIR}/tfidf_vectorizer.pkl")
emotion_labels = rf_emotion.classes_

def clean_emotion_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_emotion(text: str):
    """ทำนายอารมณ์ด้วย RandomForest + TF-IDF"""
    cleaned = clean_emotion_text(text)
    X = vectorizer_emotion.transform([cleaned])
    probs = rf_emotion.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    label = emotion_labels[pred_idx]
    confidence = float(probs[pred_idx])
    return label, confidence

def overall_risk(text_label, prob_spam, url_label, emo_label):
    score = 0.0
    score += 0.6 if text_label == "Spam" else 0.1
    if "Malicious URL" in url_label:
        score += 0.3
    if emo_label.lower() in ["fear", "anger"]:
        score += 0.1
    score = min(score, 1.0)

    if score <= 0.3:
        risk_level = "✅ Safe"
    elif score <= 0.5:
        risk_level = "⚠️ Borderline"
    elif score <= 0.7:
        risk_level = "⚠️ Suspicious"
    else:
        risk_level = "🚨 High Risk Spam"
    return score, risk_level

class Message(BaseModel):
    sender: str
    text: str
    url: str = None

@app.get("/")
def root():
    return {"message": "✅ FastAPI is running. Use POST /analyze for predictions."}
@app.post("/analyze")
def analyze_message_api(msg: Message):
    text_label, prob_ham, prob_spam = predict_text(msg.text)
    if not msg.url or msg.url.strip() == "" or msg.url.lower() == "(none)":
        url_label = "No URL provided"
    else:
        url_label = predict_url(msg.url)
    emo_label, emo_conf = predict_emotion(msg.text)
    risk_score, risk_level = overall_risk(text_label, prob_spam, url_label, emo_label)
    result = {
        "sender": msg.sender,
        "text": msg.text,
        "url": msg.url if msg.url else "(none)",
        "spam_label": text_label,
        "spam_prob": prob_spam,
        "url_status": url_label,
        "emotion": emo_label,
        "emo_confidence": emo_conf,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "timestamp": firestore.SERVER_TIMESTAMP  
    }
    safe_sender = msg.sender.replace(".", "_").replace("/", "_")
    db.collection("emails_data").document(safe_sender).collection("messages").add(result)
    print(f"[INFO] ✅ Saved for sender: {msg.sender}")
    
    return {
        "sender": msg.sender,
        "text": msg.text,
        "url": msg.url if msg.url else "(none)",
        "spam_ham": {"label": text_label, "ham_prob": prob_ham, "spam_prob": prob_spam},
        "url_status": url_label,
        "emotion": {"label": emo_label, "confidence": emo_conf},
        "overall": {"score": risk_score, "level": risk_level}
    }
