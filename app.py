import os
import re
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
import logging
from collections import Counter


APP_CONFIG = {
    "MAX_REVIEWS": 3000,
    "PAD_LEN": 100,
    "PRED_BATCH_SIZE": 256
}

MODEL_PATH = "models/lstm/lstm_sentiment_model.h5"
TOKENIZER_PATH = "models/lstm/lstm_tokenizer.pkl"

app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=logging.INFO)


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

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(TOKENIZER_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

def scrape_reviews(app_id, lang="en", max_reviews=None):
    n_reviews = int(max_reviews or APP_CONFIG["MAX_REVIEWS"])
    all_reviews = []
    token = None

    while len(all_reviews) < n_reviews:
        batch_count = min(200, n_reviews - len(all_reviews))
        batch, token = gp_reviews(
            app_id,
            lang=lang,
            count=batch_count,
            sort=gpSort.NEWEST,
            continuation_token=token
        )
        if not batch:
            break

        all_reviews.extend(batch)

        if not token:
            break

    if not all_reviews:
        return pd.DataFrame()

    df = pd.DataFrame(all_reviews)
    if "content" not in df.columns:
        df["content"] = df.get("review", "")
    if "score" not in df.columns:
        df["score"] = df.get("rating", np.nan)

    return df[["content", "score"]]

def predict_sentiments_df(df):
    df = df.copy()
    df["content"] = df["content"].astype(str)
    df["clean"] = df["content"].apply(clean_text)

    seq = tokenizer.texts_to_sequences(df["clean"].tolist())
    pad = pad_sequences(seq, maxlen=APP_CONFIG["PAD_LEN"], padding="post")

    preds = model.predict(pad, batch_size=APP_CONFIG["PRED_BATCH_SIZE"], verbose=0)
    labels = np.argmax(preds, axis=1)

    df["pred_label"] = labels
    df["sentiment"] = df["pred_label"].map({0: "Negative", 1: "Neutral", 2: "Positive"})
    df["sentiment"].replace({"Neutral": "Positive"}, inplace=True)

    return df

RECOMMENDATION_RULES = {
    "crash": "Improve app stability and fix frequent crashes.",
    "crashing": "Improve app stability and fix frequent crashes.",
    "bug": "Focus on bug fixes to improve overall reliability.",
    "buggy": "Focus on bug fixes to improve overall reliability.",
    "slow": "Optimize performance and reduce loading times.",
    "lag": "Improve responsiveness and reduce lag issues.",
    "delay": "Reduce delays and improve performance.",
    "ad": "Reduce ad frequency or offer an ad-free option.",
    "ads": "Reduce ad frequency or offer an ad-free option.",
    "advertisement": "Reduce ad frequency or offer an ad-free option.",
    "login": "Fix login and authentication issues.",
    "sign": "Fix login and authentication issues.",
    "update": "Ensure updates are stable and well-tested.",
    "support": "Improve customer support and issue resolution.",
    "ui": "Review UI/UX design based on user feedback.",
    "interface": "Improve interface clarity and usability."
}


def extract_keywords(texts, top_n=20):
    words = []
    for t in texts:
        if not isinstance(t, str):
            continue
        words.extend(t.lower().split())
    return [w for w, _ in Counter(words).most_common(top_n)]

def generate_recommendations(df_pred):
    recommendations = set()

    negative_reviews = df_pred[df_pred["sentiment"] == "Negative"]["clean"].tolist()
    positive_reviews = df_pred[df_pred["sentiment"] == "Positive"]["content"].tolist()

    neg_keywords = extract_keywords(negative_reviews)
    pos_keywords = extract_keywords(positive_reviews)

    for kw in neg_keywords:
        if kw in RECOMMENDATION_RULES:
            recommendations.add(RECOMMENDATION_RULES[kw])

    neg_ratio = len(negative_reviews) / max(len(df_pred), 1)

    if neg_ratio < 0.2:
        recommendations.add("Maintain features that users frequently praise.")


    if not recommendations:
        recommendations.add("Overall user sentiment is stable. Continue monitoring feedback.")

    return list(recommendations)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search_app")
def search_app():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    try:
        results = gp_search(q, lang="en", country="us")[:10]
    except:
        results = []

    out = []
    for r in results:
        if not r.get("appId"):
            continue
        out.append({
            "name": r.get("title", ""),
            "id": r.get("appId", ""),
            "developer": r.get("developer", ""),    
            "icon": r.get("icon", ""),
            "score": r.get("score", 0)
        })

    return jsonify(out)

@app.route("/analyze", methods=["POST"])
def analyze():
    app_id = request.form.get("selected_app_id", "").strip()
    if not app_id:
        return render_template("index.html", error="Please select an app.")

    df = scrape_reviews(app_id, max_reviews=APP_CONFIG["MAX_REVIEWS"])
    if df.empty:
        return render_template("index.html", error="No reviews found for this app.")

    df_pred = predict_sentiments_df(df)

    recommendations = generate_recommendations(df_pred)

    counts = df_pred["sentiment"].value_counts().to_dict()
    total = sum(counts.values()) or 1

    data = {
        "Positive": round((counts.get("Positive", 0) / total) * 100, 2),
        "Negative": round((counts.get("Negative", 0) / total) * 100, 2)
    }
    star_counts = df["score"].value_counts().to_dict()
    ratings = [int(star_counts.get(i, 0)) for i in [1, 2, 3, 4, 5]]

    positive_df = df_pred[df_pred["sentiment"] == "Positive"]
    negative_df = df_pred[df_pred["sentiment"] == "Negative"]

    positive_reviews = positive_df["content"].sample(
        n=min(10, len(positive_df)),
        random_state=None
    ).tolist()

    negative_reviews = negative_df["content"].sample(
        n=min(10, len(negative_df)),
        random_state=None
    ).tolist()

    samples = {
        "Positive": positive_reviews,
        "Negative": negative_reviews
    }


    return render_template(
        "dashboard.html",
        app_id=app_id,
        data=data,
        ratings=ratings,
        avg_rating=round(float(df["score"].mean()), 2),
        samples=samples,
        recommendations=recommendations
    )

@app.route("/search")
def search():
    return render_template("search.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
