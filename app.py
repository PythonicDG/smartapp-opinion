import os
import re
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google_play_scraper import reviews as gp_reviews, Sort as gpSort, search as gp_search
from google_play_scraper import app as gp_app
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from collections import Counter
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


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

def calculate_app_health_score(df, avg_rating):
    rating_score = (avg_rating / 5) * 40

    neg_ratio = len(df[df["sentiment"] == "Negative"]) / max(len(df), 1)
    negative_score = max(0, (1 - neg_ratio)) * 30

    crash_keywords = ["crash", "crashing", "freeze", "stuck", "bug"]
    crash_count = sum(
        df["clean"].str.contains(rf"\b{k}\b", regex=True).sum()
        for k in crash_keywords
    )
    crash_ratio = crash_count / max(len(df), 1)
    crash_score = max(0, (1 - crash_ratio * 3)) * 20

    recent = df.head(int(len(df) * 0.3))
    old = df.tail(int(len(df) * 0.3))

    recent_neg = len(recent[recent["sentiment"] == "Negative"]) / max(len(recent), 1)
    old_neg = len(old[old["sentiment"] == "Negative"]) / max(len(old), 1)

    trend_score = 10 if recent_neg <= old_neg else 4

    final_score = rating_score + negative_score + crash_score + trend_score

    return int(min(max(final_score, 0), 100))

def format_installs(installs):
    if not isinstance(installs, str):
        return "N/A"

    num = installs.replace("+", "").replace(",", "")
    if not num.isdigit():
        return installs

    n = int(num)

    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B+"
    elif n >= 1_000_000:
        return f"{n // 1_000_000}M+"
    elif n >= 1_000:
        return f"{n // 1_000}K+"
    else:
        return installs

ISSUE_KEYWORDS = {
    "Bug": [
        "bug", "buggy", "crash", "crashing", "freeze", "error", "issue", "problem"
    ],
    "Performance": [
        "slow", "lag", "laggy", "delay", "hang", "loading", "performance"
    ],
    "UI": [
        "ui", "interface", "design", "layout", "screen", "button", "ux"
    ],
    "Feature": [
        "feature", "add", "request", "need", "wish", "missing", "support"
    ]
}

def tag_issues(df_pred):
    """
    Returns issue distribution percentages from negative reviews
    """
    negative_reviews = df_pred[df_pred["sentiment"] == "Negative"]["clean"]

    issue_counts = {k: 0 for k in ISSUE_KEYWORDS}

    for text in negative_reviews:
        for issue, keywords in ISSUE_KEYWORDS.items():
            if any(re.search(rf"\b{kw}\b", text) for kw in keywords):
                issue_counts[issue] += 1
                break

    total = sum(issue_counts.values()) or 1

    issue_percentages = {
        k: round((v / total) * 100, 2)
        for k, v in issue_counts.items()
    }

    return issue_percentages


def get_monthly_review_volume(app_id, lang="en", max_reviews=None):
    """Fetch reviews only for monthly volume stats without touching existing scrape logic.

    Uses google_play_scraper directly so we don't modify scrape_reviews,
    and returns a dict mapping YYYY-MM -> count.
    """
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
        return {}

    df = pd.DataFrame(all_reviews)
    if "at" not in df.columns:
        return {}

    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df = df.dropna(subset=["at"])
    if df.empty:
        return {}

    df["month"] = df["at"].dt.to_period("M").astype(str)
    counts = df["month"].value_counts().sort_index()
    return counts.to_dict()


def build_comparison_profile(app_id):
    """Build a compact analytics profile for comparison dashboard for a single app."""
    # App metadata
    try:
        app_details = gp_app(app_id, lang="en", country="us")
        if not isinstance(app_details, dict):
            raise ValueError("google_play_scraper.app did not return dict")
    except Exception as e:
        logging.error(f"Failed to fetch app metadata for comparison: {e}")
        app_details = {}

    # Reviews + predictions
    df = scrape_reviews(app_id, max_reviews=APP_CONFIG["MAX_REVIEWS"])
    if df.empty:
        raise ValueError("No reviews found for this app.")

    df_pred = predict_sentiments_df(df)

    # Sentiment percentages
    counts = df_pred["sentiment"].value_counts().to_dict()
    total = sum(counts.values()) or 1
    sentiment = {
        "positive_pct": round((counts.get("Positive", 0) / total) * 100, 2),
        "negative_pct": round((counts.get("Negative", 0) / total) * 100, 2),
    }

    # Rating distribution
    star_counts = df["score"].value_counts().to_dict()
    ratings = [int(star_counts.get(i, 0)) for i in [1, 2, 3, 4, 5]]

    # Issues from negative reviews
    issues = tag_issues(df_pred)

    # Review count & avg length
    review_count = int(len(df))
    if review_count:
        word_counts = df["content"].astype(str).str.split().str.len()
        avg_review_length = float(round(word_counts.mean(), 2))
    else:
        avg_review_length = 0.0

    # Health score
    avg_rating = round(float(df["score"].mean()), 2)
    health_score = int(calculate_app_health_score(df_pred, avg_rating))

    # Monthly volume (separate fetch to avoid touching scrape_reviews)
    monthly_volume = get_monthly_review_volume(
        app_id, max_reviews=APP_CONFIG["MAX_REVIEWS"]
    )

    app_meta = {
        "title": app_details.get("title", "Title Not Available"),
        "developer": app_details.get("developer", "Developer Not Available"),
        "genre": app_details.get("genre", ""),
        "icon": app_details.get("icon", ""),
        "released": app_details.get("released", ""),
        "version": app_details.get("version", ""),
        "score": round(float(app_details.get("score", 0) or 0), 1),
        "installs": format_installs(app_details.get("installs", "N/A")),
    }

    return {
        "meta": app_meta,
        "sentiment": sentiment,
        "ratings": ratings,
        "issues": issues,
        "review_count": review_count,
        "avg_review_length": avg_review_length,
        "health_score": health_score,
        "monthly_volume": monthly_volume,
    }


def generate_comparison_verdict(app_a, app_b):
    """Generate a natural language verdict comparing two apps."""
    name_a = app_a["meta"].get("title", "App A")
    name_b = app_b["meta"].get("title", "App B")

    health_a = app_a["health_score"]
    health_b = app_b["health_score"]
    pos_a = app_a["sentiment"].get("positive_pct", 0)
    pos_b = app_b["sentiment"].get("positive_pct", 0)

    # Treat Bug + Performance as stability / technical complaints
    neg_issue_a = app_a["issues"].get("Bug", 0) + app_a["issues"].get("Performance", 0)
    neg_issue_b = app_b["issues"].get("Bug", 0) + app_b["issues"].get("Performance", 0)

    parts = []

    if health_a > health_b:
        parts.append(
            f"{name_a} outperforms {name_b} in overall app health, "
            "combining user satisfaction and stability."
        )
    elif health_b > health_a:
        parts.append(
            f"{name_b} outperforms {name_a} in overall app health, "
            "combining user satisfaction and stability."
        )
    else:
        parts.append(
            f"{name_a} and {name_b} show a similar overall health profile based on user reviews."
        )

    if pos_a > pos_b:
        parts.append(f"Users leave a higher share of positive reviews for {name_a}.")
    elif pos_b > pos_a:
        parts.append(f"Users leave a higher share of positive reviews for {name_b}.")
    else:
        parts.append("Both apps have a comparable level of positive sentiment.")

    if neg_issue_a < neg_issue_b:
        parts.append(
            f"{name_a} receives fewer bug and performance related complaints, "
            f"while {name_b} shows a higher concentration of technical issues."
        )
    elif neg_issue_b < neg_issue_a:
        parts.append(
            f"{name_b} receives fewer bug and performance related complaints, "
            f"while {name_a} shows a higher concentration of technical issues."
        )
    else:
        parts.append("Bug and performance complaint levels appear similar across both apps.")

    return " ".join(parts)


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

    try:
        app_details = gp_app(app_id, lang="en", country="us")
        if not isinstance(app_details, dict):
            raise ValueError("google_play_scraper.app did not return dict")
    except Exception as e:
        logging.error(f"Failed to fetch app metadata: {e}")
        app_details = {}


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

    issue_distribution = tag_issues(df_pred)


    avg_rating = round(float(df["score"].mean()), 2)
    health_score = calculate_app_health_score(df_pred, avg_rating)

    app_meta = {
    "title": app_details.get("title", "Title Not Available"),
    "developer": app_details.get("developer", "developer Not Available"),
    "genre": app_details.get("genre", ""),
    "icon": app_details.get("icon", ""),
    "released": app_details.get("released", ""),
    "version": app_details.get("version", ""),
    "score": round(float(app_details.get("score", 0)), 1),
    "installs": format_installs(app_details.get("installs", "N/A"))

        }

    return render_template(
        "dashboard.html",
        app_meta=app_meta,
        app_id=app_id,
        data=data,
        ratings=ratings,
        avg_rating=round(float(df["score"].mean()), 2),
        health_score=health_score,
        samples=samples,
        recommendations=recommendations,
        issue_distribution=issue_distribution
    )

@app.route("/download-report")
def download_report():
    app_id = request.args.get("app_id")
    if not app_id:
        return "Missing app_id", 400

    # Fetch reviews
    df = scrape_reviews(app_id, max_reviews=APP_CONFIG["MAX_REVIEWS"])
    if df.empty:
        return "No reviews found", 404

    df_pred = predict_sentiments_df(df)

    # App metadata
    try:
        app_details = gp_app(app_id, lang="en", country="us")
    except Exception:
        app_details = {}

    app_name = app_details.get("title", app_id)
    genre = app_details.get("genre", "N/A")
    released = app_details.get("released", "N/A")

    # Metrics
    avg_rating = round(float(df["score"].mean()), 2)
    pos_pct = round(
        (len(df_pred[df_pred["sentiment"] == "Positive"]) / len(df_pred)) * 100, 2
    )
    neg_pct = round(
        (len(df_pred[df_pred["sentiment"] == "Negative"]) / len(df_pred)) * 100, 2
    )

    health_score = calculate_app_health_score(df_pred, avg_rating)

    # Recommendations & issue tagging
    recommendations = generate_recommendations(df_pred)
    issue_distribution = tag_issues(df_pred)

    # Sample reviews
    pos_samples = (
        df_pred[df_pred["sentiment"] == "Positive"]["content"]
        .head(5)
        .tolist()
    )
    neg_samples = (
        df_pred[df_pred["sentiment"] == "Negative"]["content"]
        .head(5)
        .tolist()
    )

    # PDF generation
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "SMARTVIEW App Health Report")

    pdf.setFont("Helvetica", 11)
    y -= 30
    pdf.drawString(50, y, f"App Name: {app_name}")

    y -= 18
    pdf.drawString(50, y, f"Genre: {genre}")

    y -= 18
    pdf.drawString(50, y, f"Released: {released}")

    y -= 18
    pdf.drawString(50, y, f"Health Score: {health_score} / 100")

    y -= 18
    pdf.drawString(50, y, f"Positive Reviews: {pos_pct}%")

    y -= 18
    pdf.drawString(50, y, f"Negative Reviews: {neg_pct}%")

    # AI Recommendations
    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "AI-Based Recommendations:")

    pdf.setFont("Helvetica", 11)
    for rec in recommendations:
        y -= 16
        pdf.drawString(60, y, f"- {rec}")

    # Issue Tagging
    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Automatic Issue Tagging:")

    pdf.setFont("Helvetica", 11)
    for issue, pct in issue_distribution.items():
        y -= 16
        pdf.drawString(60, y, f"- {issue}: {pct}%")

    # Positive Samples
    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Sample Positive Reviews:")

    pdf.setFont("Helvetica", 10)
    for r in pos_samples:
        y -= 14
        pdf.drawString(60, y, f"- {r[:120]}")

    # Negative Samples
    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Sample Negative Reviews:")

    pdf.setFont("Helvetica", 10)
    for r in neg_samples:
        y -= 14
        pdf.drawString(60, y, f"- {r[:120]}")

    pdf.showPage()
    pdf.save()

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{app_id}_report.pdf",
        mimetype="application/pdf"
    )


@app.route("/export-reviews")
def export_reviews():
    app_id = request.args.get("app_id")
    if not app_id:
        return "Missing app_id", 400

    df = scrape_reviews(app_id, max_reviews=APP_CONFIG["MAX_REVIEWS"])
    if df.empty:
        return "No reviews found", 404

    df_pred = predict_sentiments_df(df)

    output = io.StringIO()
    df_pred[["content", "score", "sentiment"]].to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{app_id}_reviews.csv"
    )

@app.route("/search")
def search():
    return render_template("search.html")


@app.route("/compare", methods=["POST"])
def compare():
    """Compare two apps side by side using existing analysis pipeline."""
    app_id_1 = request.form.get("app_id_1", "").strip()
    app_id_2 = request.form.get("app_id_2", "").strip()

    if not app_id_1 or not app_id_2:
        return render_template("search.html", error="Please select two apps to compare." )

    if app_id_1 == app_id_2:
        return render_template("search.html", error="Please choose two different apps for comparison.")

    try:
        app_a_profile = build_comparison_profile(app_id_1)
        app_b_profile = build_comparison_profile(app_id_2)
    except ValueError as ve:
        return render_template("search.html", error=str(ve))
    except Exception as e:
        logging.error(f"Error during comparison: {e}")
        return render_template("search.html", error="Failed to compare the selected apps.")

    verdict = generate_comparison_verdict(app_a_profile, app_b_profile)

    # Decide winner for badge (based primarily on health score)
    winner = None
    if app_a_profile["health_score"] > app_b_profile["health_score"]:
        winner = "app_a"
    elif app_b_profile["health_score"] > app_a_profile["health_score"]:
        winner = "app_b"

    comparison_data = {
        "app_a": app_a_profile,
        "app_b": app_b_profile,
        "verdict": verdict,
        "winner": winner,
    }

    return render_template(
        "compare_dashboard.html",
        comparison_data=comparison_data,
        app_a=app_a_profile,
        app_b=app_b_profile,
        verdict=verdict,
        winner=winner,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
