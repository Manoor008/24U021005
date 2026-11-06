# app.py  --  News Headline Generator (Flask)
#
# Default: downloads a model already fine-tuned on XSum (no training needed).
# To serve your own fine-tuned T5 instead:
#   Windows (cmd):  set MODEL_NAME=./t5-headline-mini && python app.py
#   Mac/Linux:      MODEL_NAME=./t5-headline-mini python app.py
import os
import torch
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set MODEL_NAME to force one model; otherwise the first one that loads is used
_forced = os.environ.get("MODEL_NAME")
CANDIDATES = [_forced] if _forced else [
    "sshleifer/distilbart-xsum-12-3",
    "facebook/bart-large-xsum",
    "google/pegasus-xsum",
]
MAX_SOURCE_LEN = 512
MAX_NEW_TOKENS = 48
MIN_ARTICLE_CHARS = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = model = MODEL_NAME = None
for _name in CANDIDATES:
    try:
        tokenizer = AutoTokenizer.from_pretrained(_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(_name).to(device).eval()
        MODEL_NAME = _name
        break
    except Exception as e:
        print(f"Could not load {_name}: {type(e).__name__}: {str(e)[:200]}")
if model is None:
    raise SystemExit("No model could be loaded; see the errors above.")
print("Serving model:", MODEL_NAME)

# T5 needs the task prefix; BART/PEGASUS-style models do not
PREFIX = "summarize: " if model.config.model_type == "t5" else ""

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>News Headline Generator</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
  textarea { width: 100%; height: 220px; padding: 10px; font-size: 15px; box-sizing: border-box; }
  button { margin-top: 10px; padding: 10px 22px; font-size: 15px; cursor: pointer; }
  #out { margin-top: 24px; font-size: 22px; font-weight: 600; }
  small { color: #666; }
</style>
</head>
<body>
  <h2>Automatic News Headline Generator</h2>
  <small>Model: {{ model_name }}</small>
  <p>Paste a news article:</p>
  <textarea id="article"></textarea><br>
  <button onclick="gen()">Generate headline</button>
  <div id="out"></div>
<script>
async function gen() {
  const out = document.getElementById("out");
  out.textContent = "Generating...";
  const r = await fetch("/generate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({article: document.getElementById("article").value})
  });
  const d = await r.json();
  out.textContent = d.headline || d.error;
}
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE, model_name=MODEL_NAME)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": device, "model": MODEL_NAME})


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    article = (data.get("article") or "").strip()

    if len(article) < MIN_ARTICLE_CHARS:
        return jsonify({"error": f"'article' is required (min {MIN_ARTICLE_CHARS} characters)."}), 400

    try:
        num_beams = int(data.get("num_beams", 4))
    except (TypeError, ValueError):
        return jsonify({"error": "'num_beams' must be an integer."}), 400
    num_beams = max(1, min(num_beams, 8))

    inputs = tokenizer(
        PREFIX + article,
        return_tensors="pt",
        max_length=MAX_SOURCE_LEN,
        truncation=True,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=MAX_NEW_TOKENS,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    headline = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return jsonify({"headline": headline})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
