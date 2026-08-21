# EAPDS — Email Anti-Phishing Detection System

ระบบตรวจจับอีเมล/ข้อความฟิชชิงด้วย Machine Learning ให้บริการผ่าน REST API (FastAPI)
รวมผลจากโมเดล 3 ตัวเป็นคะแนนความเสี่ยง (risk score) เดียว และบันทึกผลลง Firebase Firestore

## ระบบทำงานยังไง

รับข้อความเข้ามาแล้ววิเคราะห์พร้อมกัน 3 ด้าน:

| ด้าน | โมเดล | ไฟล์ |
|---|---|---|
| Spam / Ham (เนื้อหาข้อความ) | DeBERTa (fine-tuned) | `models/deberta_finetuned_v579/` |
| ตรวจ URL อันตราย | LSTM (Keras) | `model_full/url_full/my_lstm_url_model.h5` |
| วิเคราะห์อารมณ์ข้อความ | RandomForest + TF-IDF | `models_emotion45/` |

จากนั้นรวมเป็น **risk score** (✅ Safe / ⚠️ Borderline / ⚠️ Suspicious / 🚨 High Risk) แล้วบันทึกผลลง Firestore

## โครงสร้างโปรเจกต์

```
EAPDS/
├── app.py                       # FastAPI backend (endpoint /analyze)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── firebase_key.example.json    # ตัวอย่าง (ต้องสร้าง firebase_key.json จริงเอง)
├── models/
│   └── deberta_finetuned_v579/  # โมเดล Spam/Ham — ต้องวางเอง (ดูด้านล่าง)
├── model_full/
│   └── url_full/                # โมเดล URL (อยู่ใน repo)
└── models_emotion45/            # โมเดล Emotion (อยู่ใน repo)
```

> **หมายเหตุ:** โมเดล DeBERTa (`model.safetensors` ~737MB) และ `firebase_key.json` **ไม่ได้อยู่ใน repo** — ตัวแรกเพราะไฟล์ใหญ่เกินลิมิต GitHub ตัวหลังเพราะเป็นความลับ ต้องเตรียมเองตามขั้นตอนด้านล่าง

## การติดตั้ง

```bash
pip install -r requirements.txt
```

### 1) เตรียม Firebase key

คัดลอก `firebase_key.example.json` เป็น `firebase_key.json` แล้วใส่ค่าจริงจาก Firebase Console
(Project Settings → Service accounts → Generate new private key)

⚠️ **ห้าม commit `firebase_key.json` ขึ้น git** — มี `.gitignore` กันไว้แล้ว

### 2) เตรียมโมเดล DeBERTa

วางไฟล์โมเดล fine-tuned ทั้งหมดไว้ที่ `models/deberta_finetuned_v579/` ให้มีไฟล์ครบ:
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `added_tokens.json`, `spm.model`

(เก็บสำเนาโมเดลไว้เอง หรือแชร์ผ่าน Google Drive / Hugging Face แล้วดาวน์โหลดมาวาง)

## การรัน

```bash
# รันตรง
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# หรือใช้ Docker
docker compose up --build
```

## การใช้งาน API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "hr@example.com",
    "text": "Please verify your account now",
    "url": "http://suspicious-link.com/login"
  }'
```

ตัวอย่างผลลัพธ์:

```json
{
  "sender": "hr@example.com",
  "spam_ham": { "label": "Spam", "ham_prob": 0.12, "spam_prob": 0.88 },
  "url_status": "Malicious URL (conf=0.91)",
  "emotion": { "label": "fear", "confidence": 0.74 },
  "overall": { "score": 0.9, "level": "🚨 High Risk Spam" }
}
```
