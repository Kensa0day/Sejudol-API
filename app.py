import os
import re
import pymysql
import numpy as np
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from joblib import load as joblib_load
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from crawler import crawl_and_return
import asyncio
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')

# Load .env
load_dotenv()
# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_USER = os.getenv("DB_USER", "root")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "")
# DB_NAME = os.getenv("DB_NAME", "sentiment_db")

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    # Connect ke Neon via DATABASE_URL (dengan pgbouncer)
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# di awal file
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_API_URL          = "https://api-inference.huggingface.co/models/w11wo/indonesian-roberta-base-sentiment-classifier"
HEADERS             = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# App
app = Flask(__name__)
CORS(app)

# Database Connection
# def get_db_connection():
#     return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME)

# Load Models
# NB: tfidf_vectorizer dipakai hanya untuk Naive Bayes
tfidf_vectorizer = joblib_load("tfidf_vectorizer.joblib")
nb_model = joblib_load("naive_bayes_model.joblib")

# SVM di‐pack sebagai pipeline (TF-IDF di dalamnya), jadi kita cukup load file pipeline saja
svm_pipeline = joblib_load("pipeline_svm_best.joblib")
# svm_pipeline = joblib_load("pipeline_svm_all.joblib")

# Load IndoBERT
# MODEL_PATH = "my-finetuned-bert"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
# bert_pipeline = pipeline("text-classification", model=bert_model, tokenizer=tokenizer)

# Risiko Keywords
RISK_KEYWORDS = ["slot", "judol", "ngeslot", "pinjol", "gacor", "jp", "depo", "spin", "wd", "jackpot", "togel"]

# Preprocessing Tools
kamus_data = pd.read_excel('kamuskatabaku.xlsx')
kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@[\w_]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def normalize_text(text):
    return ' '.join([kamus_tidak_baku.get(word, word) for word in text.split()])

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stem_words(tokens):
    return ' '.join([stemmer.stem(word) for word in tokens])

def detect_risk(text):
    """
    Mengembalikan True jika salah satu kata di RISK_KEYWORDS muncul di dalam teks.
    Pastikan teks sudah di‐lowercase sebelum cek.
    """
    return any(k in text for k in RISK_KEYWORDS)

# ──────────────────────────────────────────────────────────────────────────────
# Crawl Tweet Endpoint
@app.route("/crawl-tweets", methods=["POST"])
def crawl_tweets():
    data = request.get_json() or {}
    jumlah = int(data.get("jumlah", 50))
    try:
        # Ambil tweets (cuma text + created_at)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tweets = loop.run_until_complete(crawl_and_return(jumlah))

        inserted = 0
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for tweet in tweets:
                    teks    = tweet["text"]
                    created = tweet["created_at"]

                    # 1) Hitung risk_flag berdasarkan teks mentah (atau bisa juga pake clean_text)
                    cleaned_for_risk = clean_text(teks)
                    is_risky = 1 if detect_risk(cleaned_for_risk) else 0

                    # 2) Lakukan INSERT, termasuk kolom risk_flag
                    cur.execute("""
                        INSERT INTO comments (text, created_at, risk_flag)
                        VALUES (%s, %s, %s)
                    """, (teks, created, is_risky))
                    inserted += 1

                conn.commit()

        return jsonify({"status": "success", "inserted": inserted}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/preprocess", methods=["POST"])
def preprocess_with_hf_api():
    conn   = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("""
        SELECT id, text
          FROM comments
         WHERE sentiment IS NULL
            OR sentiment = ''
            OR sentiment = 'unknown'
    """)
    rows = cursor.fetchall()
    app.logger.info(f"📝 Found {len(rows)} rows to process")

    total = 0
    for row in rows:
        cid, raw_text = row["id"], row["text"]
        try:
            # 1) Clean & normalize
            cleaned    = clean_text(raw_text)
            normalized = normalize_text(cleaned)

            # 2) Prepare stemmed for DB storage
            tokens     = normalized.split()
            filtered   = remove_stopwords(tokens)
            stemmed    = " ".join(stem_words(filtered))

            # 3) Call HF Inference API (use normalized text)
            # 3) Call HF Inference API (use normalized text)
            payload = {"inputs": normalized, "options": {"wait_for_model": True}}
            resp    = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
            resp.raise_for_status()
            result  = resp.json()

            # 4) Normalize the nested list/dict response into a single dict `hf_obj`
            hf_obj = None
            if isinstance(result, list) and result:
                first = result[0]
                # Jika first adalah list, ambil elemen pertamanya
                if isinstance(first, list) and first:
                    hf_obj = first[0]
                # Jika first sudah dict
                elif isinstance(first, dict):
                    hf_obj = first

            # 5) Map API response to our labels
            sentiment  = "unknown"
            confidence = 0.0
            if isinstance(hf_obj, dict):
                lbl = hf_obj.get("label", "").lower()
                sc  = hf_obj.get("score", 0.0)
                confidence = float(sc)
                if "positive" in lbl:
                    sentiment = "positif"
                elif "negative" in lbl:
                    sentiment = "negatif"
                elif "neutral" in lbl:
                    sentiment = "netral"


            # 5) Risk flag
            risk_flag = 1 if detect_risk(cleaned) else 0

            # 6) Update database
            cursor.execute("""
                UPDATE comments
                   SET cleaning    = %s,
                       sentiment   = %s,
                       confidence  = %s,
                       risk_flag   = %s
                 WHERE id = %s
            """, (stemmed, sentiment, confidence, risk_flag, cid))

            total += 1

        except Exception as e:
            app.logger.error(f"Error processing id={cid}: {e}", exc_info=True)
            # skip this row
            continue

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"status": "success", "total_processed": total}), 200




    
# @app.route("/preprocess", methods=["POST"])
# def preprocess_svm_only_overwrite():
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor(pymysql.cursors.DictCursor)

#         # Ambil baris yang belum diproses SVM (sentiment IS NULL atau 'unknown')
#         cursor.execute("""
#             SELECT id, text 
#               FROM comments 
#              WHERE sentiment IS NULL 
#                 OR sentiment = 'unknown'
#         """)
#         rows = cursor.fetchall()

#         if not rows:
#             cursor.close()
#             conn.close()
#             return jsonify({
#                 "status": "success",
#                 "total_processed": 0
#             }), 200

#         total = 0
#         for row in rows:
#             cid   = row["id"]
#             text  = row["text"]

#             # ——— 1) PREPROCESSING ———
#             cleaned_basic = clean_text(text)
#             normalized    = normalize_text(cleaned_basic)
#             tokens        = normalized.split()
#             filtered      = remove_stopwords(tokens)
#             stemmed       = stem_words(filtered)

#             # ——— 2) Prediksi SVM via pipeline (cukup kirim list of str) ———
#             # pipeline otomatis melakukan TF-IDF pada data yang dipreprocess sama seperti training
#             raw_pred = svm_pipeline.predict([stemmed])[0]
#             # Hitung confidence (pakai decision_function)
#             decision = svm_pipeline.decision_function([stemmed])[0]
#             if isinstance(decision, np.ndarray):
#                 score = max(decision)
#             else:
#                 score = decision
#             conf     = float(1.0 / (1.0 + np.exp(-score)))

#             # ——— 3) Mapping raw_pred ke label string ———
#             # Jika pipeline SVM dikonfigurasi untuk output angka: {0,1,2}
#             if isinstance(raw_pred, (int, np.integer)):
#                 label_map = {0: "negatif", 1: "netral", 2: "positif"}
#                 label_svm = label_map.get(int(raw_pred), "unknown")
#             else:
#                 # Jika pipeline mengeluarkan label string langsung ("negatif"/"positif"/…)
#                 label_svm = raw_pred

#             if not label_svm:
#                 label_svm = "unknown"

#             # ——— 4) Hitung risk_flag berdasarkan teks yang sudah dibersihkan ———
#             is_risky = 1 if detect_risk(cleaned_basic) else 0

#             # ——— 5) UPDATE database (cleaning, sentiment, confidence, risk_flag) ———
#             cursor.execute("""
#                 UPDATE comments
#                    SET cleaning    = %s,
#                        sentiment   = %s,
#                        confidence  = %s,
#                        risk_flag   = %s
#                  WHERE id = %s
#             """, (stemmed, label_svm, conf, is_risky, cid))

#             total += 1

#         conn.commit()
#         cursor.close()
#         conn.close()

#         return jsonify({
#             "status": "success",
#             "total_processed": total
#         }), 200

#     except Exception as e:
#         return jsonify({"status": "error", "error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# Predict Endpoint
# … kode impor, inisialisasi Flask, load model, dan fungsi-fungsi preprocessing di atas tetap sama …

# ──────────────────────────────────────────────────────────────────────────────
# Predict Endpoint
@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.get_json() or {}
    comment = data.get("komentar", "")

    if not comment or not isinstance(comment, str):
        return jsonify({"error": "Komentar tidak valid"}), 400

    # 1) Preprocessing
    cleaned_basic = clean_text(comment)
    normalized    = normalize_text(cleaned_basic)
    tokens        = normalized.split()
    filtered      = remove_stopwords(tokens)
    stemmed       = stem_words(filtered)

    # 2) Naive Bayes
    X_tfidf = tfidf_vectorizer.transform([cleaned_basic])
    pred_nb = nb_model.predict(X_tfidf)[0]
    conf_nb = float(nb_model.predict_proba(X_tfidf).max())

    # 3) SVM
    raw_pred_svm = svm_pipeline.predict([stemmed])[0]
    decision     = svm_pipeline.decision_function([stemmed])[0]
    score_svm    = float(max(decision)) if isinstance(decision, np.ndarray) else float(decision)
    conf_svm     = 1.0 / (1.0 + np.exp(-score_svm))

    # 4) Hugging Face Inference API
    try:
        resp = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json={
            "inputs": [normalized],
            "options": {"wait_for_model": True}
        }, timeout=60
        )

        app.logger.info(f"HF API status={resp.status_code}, body={resp.text}")
        resp.raise_for_status()
       
        result = resp.json()
        # bisa jadi [[{...}]]
        if isinstance(result, list) and result and isinstance(result[0], list):
            hf_obj = result[0][0]
        elif isinstance(result, list) and result:
            hf_obj = result[0]
        else:
            hf_obj = {}

        lbl = hf_obj.get("label", "").lower()
        sc  = hf_obj.get("score", 0.0)
        hf_conf = float(sc)
        if "positive" in lbl:
            hf_sentiment = "positif"
        elif "negative" in lbl:
            hf_sentiment = "negatif"
        elif "neutral" in lbl:
            hf_sentiment = "netral"
        else:
            hf_sentiment = "unknown"
        
    except Exception as e:
        app.logger.error(f"HF Inference error: {e}", exc_info=True)
        hf_sentiment = "unknown"
        hf_conf = 0.0

    # 5) Risk flag
    risk_flag_val = 1 if detect_risk(cleaned_basic) else 0

    # 6) Mapping Naive Bayes label
    if isinstance(pred_nb, (int, np.integer)):
        nb_label_map = {0: "negatif", 1: "netral", 2: "positif"}
        nb_sentiment = nb_label_map.get(int(pred_nb), "unknown")
    else:
        nb_sentiment = str(pred_nb)

    # 7) Mapping SVM label
    if isinstance(raw_pred_svm, (int, np.integer)):
        svm_label_map = {0: "negatif", 1: "netral", 2: "positif"}
        svm_sentiment = svm_label_map.get(int(raw_pred_svm), "unknown")
    else:
        svm_sentiment = str(raw_pred_svm)

    # 8) Kembalikan semua hasil
    return jsonify({
        "input": comment,
        "naive_bayes": {
            "sentiment": nb_sentiment,
            "confidence": round(conf_nb, 4)
        },
        "svm": {
            "sentiment": svm_sentiment,
            "confidence": round(conf_svm, 4)
        },
        "huggingface": {
            "sentiment": hf_sentiment,
            "confidence": round(hf_conf, 4)
        },
        "risk_flag": risk_flag_val
    }), 200


# … sisanya (get_comments, delete, dll.) tetap sama …

# ──────────────────────────────────────────────────────────────────────────────
# Get Comments (termasuk risk_flag)
@app.route("/get-comments", methods=["GET"])
def get_comments():
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT id,
                       text,
                       cleaning,
                       sentiment,
                       confidence,
                       risk_flag,
                       created_at
                  FROM comments
              ORDER BY created_at DESC
                 LIMIT 10000
            """)
            rows = cur.fetchall()
    return jsonify(rows)

# ──────────────────────────────────────────────────────────────────────────────
# Delete Comment
@app.route('/delete-comment/<int:comment_id>', methods=['DELETE'])
def delete_comment(comment_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM comments WHERE id = %s", (comment_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"status": "error", "error": "ID tidak ditemukan"}), 404

        cursor.execute("DELETE FROM comments WHERE id = %s", (comment_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"status": "success", "deleted_id": comment_id})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# Risk Detection Only
@app.route("/detect-risk", methods=["POST"])
def risk_only():
    data = request.get_json() or {}
    comment = data.get("komentar", "")
    cleaned_basic = clean_text(comment)
    return jsonify({"risk_flag": 1 if detect_risk(cleaned_basic) else 0})

# ──────────────────────────────────────────────────────────────────────────────
@app.route('/deleteall-comments', methods=['DELETE'])
def delete_all_comments():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM comments")
            deleted = cursor.rowcount
            conn.commit()
        conn.close()
        return jsonify({
            "status": "success",
            "message": f"{deleted} komentar berhasil dihapus."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Run Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
