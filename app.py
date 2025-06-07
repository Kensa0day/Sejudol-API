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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from crawler import crawl_and_return
import asyncio
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')

# Load .env
load_dotenv()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "sentiment_db")

# App
app = Flask(__name__)
CORS(app)

# Database Connection
def get_db_connection():
    return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME)

# Load Models
# NB: tfidf_vectorizer dipakai hanya untuk Naive Bayes
tfidf_vectorizer = joblib_load("tfidf_vectorizer.joblib")
nb_model = joblib_load("naive_bayes_model.joblib")

# SVM di‐pack sebagai pipeline (TF-IDF di dalamnya), jadi kita cukup load file pipeline saja
svm_pipeline = joblib_load("pipeline_svm_best.joblib")
# svm_pipeline = joblib_load("pipeline_svm_all.joblib")

# Load IndoBERT
MODEL_PATH = "my-finetuned-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
bert_pipeline = pipeline("text-classification", model=bert_model, tokenizer=tokenizer)

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


hf_pipe = pipeline(
    "text-classification",
    model="w11wo/indonesian-roberta-base-sentiment-classifier",
    tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier",
    return_all_scores=False
)
# ──────────────────────────────────────────────────────────────────────────────
# Preprocess + Klasifikasi SVM-only (Overwrite sentiment)

@app.route("/preprocess", methods=["POST"])
def preprocess_with_huggingface_local():
    conn   = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # Ambil baris yang belum diproses
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
            cleaned    = clean_text(raw_text)                      # lower+strip urls/mentions/punct
            normalized = normalize_text(cleaned)                   # ganti slang

            # 2) Precompute stemmed only for DB storage (optional)
            stemmed    = " ".join(stem_words(remove_stopwords(normalized.split())))

            # 3) Prediksi dengan HF memakai **normalized** (bukan stemmed!)
            hf_res = hf_pipe(normalized, truncation=True)[0]       # gunakan normalized
            lbl    = hf_res["label"].lower()
            sc     = float(hf_res["score"])
            if "positive" in lbl or "label_2" in lbl:
                sentiment = "positif"
            elif "negative" in lbl or "label_0" in lbl:
                sentiment = "negatif"
            else:
                sentiment = "netral"

            # 4) Risk flag pakai cleaned
            risk_flag = 1 if detect_risk(cleaned) else 0

            # 5) Update DB: simpan stemmed teks di cleaning, bukan untuk inference
            cursor.execute("""
                UPDATE comments
                   SET cleaning    = %s,
                       sentiment   = %s,
                       confidence  = %s,
                       risk_flag   = %s
                 WHERE id = %s
            """, (stemmed, sentiment, sc, risk_flag, cid))

            total += 1

        except Exception as e:
            app.logger.error(f"Error processing id={cid}: {e}", exc_info=True)
            continue

    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"status":"success","total_processed": total}), 200



    
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
    comment = data.get("komentar")

    if not comment or not isinstance(comment, str):
        return jsonify({"error": "Komentar tidak valid"}), 400

    # 1) Preprocessing
    cleaned_basic = clean_text(comment)
    normalized    = normalize_text(cleaned_basic)       # hanya 1 argumen
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
    if isinstance(decision, np.ndarray):
        score_svm = float(max(decision))
    else:
        score_svm = float(decision)
    conf_svm = float(1.0 / (1.0 + np.exp(-score_svm)))

    # 4) IndoBERT
    bert_result    = bert_pipeline(cleaned_basic)[0]
    bert_label     = bert_result["label"]
    bert_score     = float(bert_result["score"])
    bert_label_map = {"LABEL_0": "negatif", "LABEL_1": "netral", "LABEL_2": "positif"}
    bert_sentiment = bert_label_map.get(bert_label, bert_label)
    if bert_score < 0.58:
        bert_sentiment = "netral"

    # 5) HuggingFace Roberta‐base
    hf_res = hf_pipe(normalized, truncation=True)[0]
    hf_lbl = hf_res["label"].lower()   # misal "LABEL_0" atau "positive"
    hf_sc  = float(hf_res["score"])
    if "positive" in hf_lbl or "label_2" in hf_lbl:
        hf_sentiment = "positif"
    elif "negative" in hf_lbl or "label_0" in hf_lbl:
        hf_sentiment = "negatif"
    else:
        hf_sentiment = "netral"

    # 6) Risk flag
    risk_flag_val = 1 if detect_risk(cleaned_basic) else 0

    # 7) Mapping Naive Bayes
    if isinstance(pred_nb, (int, np.integer)):
        nb_label_map = {0: "negatif", 1: "netral", 2: "positif"}
        nb_sentiment = nb_label_map.get(int(pred_nb), "unknown")
    else:
        nb_sentiment = str(pred_nb)

    # 8) Mapping SVM
    if isinstance(raw_pred_svm, (int, np.integer)):
        svm_label_map = {0: "negatif", 1: "netral", 2: "positif"}
        svm_sentiment = svm_label_map.get(int(raw_pred_svm), "unknown")
    else:
        svm_sentiment = str(raw_pred_svm)

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
        "indobert": {
            "sentiment": bert_sentiment,
            "confidence": round(bert_score, 4)
        },
        "huggingface": {
            "sentiment": hf_sentiment,
            "confidence": round(hf_sc, 4)
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
