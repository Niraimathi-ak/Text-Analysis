from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from rake_nltk import Rake  # <-- Using RAKE instead of TF-IDF

app = Flask(__name__)

# Load NLP pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# In-memory history storage (last 5 entries)
history = []


def extract_keywords(text, top_n=6):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return keywords[:top_n]


def add_to_history(entry):
    """Add a new entry to history and keep only the last 5."""
    history.append(entry)
    if len(history) > 5:
        history.pop(0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize_text():
    payload = request.get_json(force=True)
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = summarizer(text, max_length=170, min_length=30, do_sample=False, truncation=True)[0]["summary_text"]
    add_to_history({"type": "summary", "input": text, "output": summary})

    return jsonify({"summary": summary})


@app.route("/sentiment", methods=["POST"])
def sentiment():
    payload = request.get_json(force=True)
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = sentiment_analyzer(text)[0]
    add_to_history({"type": "sentiment", "input": text, "output": result})

    return jsonify({"sentiment": result})


@app.route("/keywords", methods=["POST"])
def keywords():
    payload = request.get_json(force=True)
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    kws = extract_keywords(text, top_n=8)
    add_to_history({"type": "keywords", "input": text, "output": kws})

    return jsonify({"keywords": kws})


@app.route("/analyze_all", methods=["POST"])
def analyze_all():
    payload = request.get_json(force=True)
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = summarizer(text, max_length=130, min_length=30, do_sample=False, truncation=True)[0]["summary_text"]
    sentiment = sentiment_analyzer(text)[0]
    kws = extract_keywords(text, top_n=8)

    add_to_history({
        "type": "all",
        "input": text,
        "output": {
            "summary": summary,
            "sentiment": sentiment,
            "keywords": kws
        }
    })

    return jsonify({
        "summary": summary,
        "sentiment": sentiment,
        "keywords": kws
    })


@app.route("/history", methods=["GET"])
def get_history():
    """Return the stored history."""
    return jsonify(history)


if __name__ == "__main__":
    app.run(debug=True)
