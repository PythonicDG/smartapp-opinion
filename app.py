import os, re, time, random
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google_play_scraper import reviews as gp_reviews, Sort as gpSort, search as gp_search
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask_caching import Cache

# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# ----------------------------
# App Config
# ----------------------------
APP_CONFIG = {
    "MAX_REVIEWS": 300,
    "PAD_LEN": 100,
    "T5_MODEL": "t5-small",
}

MODEL_PATH = "models/lstm/lstm_sentiment_model.h5"
TOKENIZER_PATH = "models/lstm/lstm_tokenizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError("Model/tokenizer not found.")

# Load Model + Tokenizer once
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

# ----------------------------
# Text Preprocessing
# ----------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

# ----------------------------
# Review Scraping (Optimized)
# ----------------------------
def scrape_reviews(app_id, lang="en"):
    n_reviews = 400  # Limit to 400 for speed
    all_reviews, token = [], None
    while len(all_reviews) < n_reviews:
        batch, token = gp_reviews(
            app_id,
            lang=lang,
            count=min(200, n_reviews - len(all_reviews)),
            sort=gpSort.NEWEST,
            continuation_token=token
        )
        if not batch:
            break
        all_reviews.extend(batch)
        if not token:
            break
        # Reduced sleep time for faster scraping
        time.sleep(random.uniform(0.3, 0.6))
    if not all_reviews:
        return pd.DataFrame()
    df = pd.DataFrame(all_reviews)
    return df.rename(columns={"content": "content", "score": "score"})

# ----------------------------
# Sentiment Prediction (Batched)
# ----------------------------
def predict_sentiments_df(df):
    df = df.copy()
    df["clean"] = df["content"].astype(str).apply(clean_text)
    batch_size = 100
    all_preds = []
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        seq = tokenizer.texts_to_sequences(df["clean"].iloc[start:end].tolist())
        pad = pad_sequences(seq, maxlen=APP_CONFIG["PAD_LEN"], padding="post", truncating="post")
        preds = model.predict(pad, verbose=0)
        all_preds.append(np.argmax(preds, axis=1))
    df["pred_label"] = np.concatenate(all_preds)
    df["sentiment"] = df["pred_label"].map({0: "Negative", 1: "Neutral", 2: "Positive"})
    df["sentiment"] = df["sentiment"].replace({"Neutral": "Positive"})
    return df

# ----------------------------
# Summarization
# ----------------------------
def get_summarizer():
    from transformers import pipeline
    summarizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        truncation=True
    )
    return summarizer

def summarize_list(items, mode="pros"):
    summarizer = get_summarizer()
    if not summarizer or not items:
        return None

    text = " ".join(items)
    text = text[:8000]  # Limit input size

    if mode == "pros":
        prompt = f"""
        You are an app analyst. Summarize the following positive user reviews into a concise, human-like paragraph highlighting the app's strengths and benefits:
        {text}
        """
    else:
        prompt = f"""
        You are an app analyst. Summarize the following negative user reviews into a concise, professional paragraph discussing the main issues and complaints:
        {text}
        """

    try:
        response = summarizer(
            prompt,
            max_new_tokens=120,
            temperature=0.8,
            repetition_penalty=2.0,
            top_p=0.9,
            do_sample=True,
        )
        return response[0]["generated_text"].strip()
    except Exception as e:
        app.logger.warning(f"Summarization failed: {e}")
        return None

# ----------------------------
# Flask Routes
# ----------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search_app", methods=["GET"])
def search_app():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    try:
        results = gp_search(query, lang="en", country="us")
        results = results[:10]
        suggestions = [
            {
                "name": r.get("title", ""),
                "id": r.get("appId", ""),
                "developer": r.get("developer", ""),
                "icon": r.get("icon", ""),
                "score": r.get("score", 0)
            }
            for r in results if r.get("appId")
        ]
        return jsonify(suggestions)
    except Exception as e:
        app.logger.warning(f"Search failed: {e}")
        return jsonify([])

@app.route("/analyze", methods=["POST"])
@cache.cached(timeout=120, query_string=True)
def analyze():
    app_id = request.form.get("selected_app_id", "").strip()
    if not app_id:
        return render_template("index.html", error="Please select an app.")

    df = scrape_reviews(app_id)
    if df.empty:
        return render_template("index.html", error="No reviews found for this app.")

    df_pred = predict_sentiments_df(df)
    counts = df_pred["sentiment"].value_counts().to_dict()
    total = sum(counts.values()) or 1
    data = {
        "Positive": round((counts.get("Positive", 0) / total) * 100, 2),
        "Negative": round((counts.get("Negative", 0) / total) * 100, 2)
    }

    star_counts = df["score"].value_counts().to_dict()
    ratings = [star_counts.get(i, 0) for i in [1, 2, 3, 4, 5]]
    avg_rating = round(df["score"].mean(), 2)
    samples = {
        "Positive": df_pred[df_pred["sentiment"] == "Positive"]["content"].head(5).tolist(),
        "Negative": df_pred[df_pred["sentiment"] == "Negative"]["content"].head(5).tolist()
    }

    pros = df_pred[df_pred["sentiment"] == "Positive"]["content"].head(40).tolist()
    cons = df_pred[df_pred["sentiment"] == "Negative"]["content"].head(40).tolist()

    pros_summary = summarize_list(pros, "pros")
    cons_summary = summarize_list(cons, "cons")

    return render_template(
        "dashboard.html",
        app_id=app_id,
        data=data,
        ratings=ratings,
        avg_rating=avg_rating,
        total_reviews=len(df_pred),
        samples=samples,
        pros_summary=pros_summary,
        cons_summary=cons_summary
    )

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
