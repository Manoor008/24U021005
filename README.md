# Automatic News Headline Generator — Abstractive Summarization Platform

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

**Scholar No**: `24U021005` &nbsp;|&nbsp; **Student**: Manoor Sufiyan Ansari  
**Institute**: Indian Institute of Information Technology Bhopal (IIIT Bhopal)  
**Department**: Department of Computer Science & Engineering / Information Technology  

</div>

---

An end-to-end Natural Language Processing (NLP) and Deep Learning platform for automatic news headline generation using state-of-the-art Sequence-to-Sequence (Seq2Seq) Transformer models (Google T5, DistilBART, BART Large, and PEGASUS) fine-tuned on the BBC XSum dataset. The system features an interactive Flask web application, configurable beam-search decoding, full fine-tuning pipelines in JupyterLab, and automated ROUGE benchmark evaluation.

---

## 🏛️ System Architecture

- **Core Models**:
  - `T5 (Text-to-Text Transfer Transformer)`: Fine-tuned Seq2Seq model utilizing task-specific prefixing (`"summarize: "`).
  - `DistilBART-XSum-12-3`: Lightweight distilled checkpoint for ultra-low latency real-time generation.
  - `BART-Large-XSum` & `Google PEGASUS-XSum`: Heavyweight autoregressive models for publication-grade syntactic accuracy.
- **Inference & Decoding**:
  - Configurable Beam Search (`num_beams` 1–8).
  - Length normalization and truncation control (`max_new_tokens=48`, `max_source_len=512`).
  - Strict 3-gram repetition avoidance (`no_repeat_ngram_size=3`).
- **Backend & Serving**: Python, Flask 3.0+, PyTorch (CUDA GPU & CPU fallbacks), Hugging Face `transformers` and `accelerate`.
- **Frontend**: Clean, responsive interface served directly with embedded styling.
- **Evaluation Pipeline**: ROUGE metrics (`ROUGE-1`, `ROUGE-2`, `ROUGE-L`) calculated using Hugging Face `evaluate`.

---

## 📋 Prerequisites

Ensure you have the following installed on your system:
- **Python** (v3.10 or newer)
- **pip** (Python package installer)
- **PyTorch** (configured with CUDA for GPU acceleration, or CPU fallback)
- **Git**

---

## ⚙️ Environment Configuration

Configuration options can be set via environment variables or in `news-headline-generator/.env`:

```env
# Optional: Force a specific Hugging Face model or local checkpoint folder
# Default fallback candidates:
# 1. sshleifer/distilbart-xsum-12-3
# 2. facebook/bart-large-xsum
# 3. google/pegasus-xsum
MODEL_NAME=sshleifer/distilbart-xsum-12-3

# Optional: Server port (defaults to 5000)
PORT=5000
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Manoor008/24U021005.git
cd 24U021005
```

---

### 2. Set Up the Python Virtual Environment

```bash
cd news-headline-generator

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (Command Prompt):
venv\Scripts\activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Start the Web Application

```bash
python app.py
```

- **Interactive Web Interface**: [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Health & Device Status**: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

#### Serving Your Own Fine-Tuned Checkpoint:
If you train your own model checkpoint (e.g. `./t5-headline-final`) using the training notebook:
- **Windows (CMD)**: `set MODEL_NAME=./t5-headline-final && python app.py`
- **Windows (PowerShell)**: `$env:MODEL_NAME="./t5-headline-final"; python app.py`
- **Linux / macOS**: `MODEL_NAME=./t5-headline-final python app.py`

---

### 4. Running Training & Evaluation Notebooks

To fine-tune models from scratch or run ROUGE evaluations:

```bash
# From within the news-headline-generator directory:
jupyter lab notebooks/
```

- **`t5_xsum_headline_jupyterlab.ipynb`**: Complete end-to-end fine-tuning pipeline for `t5-small` on the BBC XSum dataset with Hugging Face `Seq2SeqTrainer`, fp16 acceleration, and checkpoint saving.
- **`headline_generator_quick.ipynb`**: Rapid inference benchmark, comparing model-generated headlines side-by-side with ground-truth reference summaries, and computing ROUGE-1, ROUGE-2, and ROUGE-L metrics.

---

## 📡 REST API Documentation

### 1. `POST /generate`
Generates an abstractive headline from the supplied article text.

**Request Body (`application/json`):**
```json
{
  "article": "Researchers at the institute have demonstrated a new autonomous drone navigation system capable of operating in dense urban environments without GPS. The team utilized deep reinforcement learning and onboard visual-inertial odometry to achieve millisecond-level trajectory planning.",
  "num_beams": 4
}
```

**Response (`200 OK`):**
```json
{
  "headline": "Autonomous drone system enables GPS-free urban navigation"
}
```

### 2. `GET /health`
Returns system status, active hardware acceleration device (`cuda` or `cpu`), and the loaded model identifier.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "device": "cuda",
  "model": "sshleifer/distilbart-xsum-12-3"
}
```

---

## 📁 Project Structure

```text
24U021005/
├── .gitignore                                 # Git ignore rules for virtual environments & model weights
├── README.md                                  # Repository overview and quick start guide
│
└── news-headline-generator/                   # Core Web Application & NLP Pipeline
    ├── app.py                                 # Flask application, REST API & embedded UI
    ├── requirements.txt                       # Python dependencies (Flask, Transformers, Torch, etc.)
    ├── .env.example                           # Example environment variable configuration
    ├── README.md                              # Sub-project docs, endpoints & usage guide
    └── notebooks/                             # Jupyter notebooks for model training & evaluation
        ├── t5_xsum_headline_jupyterlab.ipynb  # T5-small fine-tuning pipeline on XSum dataset
        └── headline_generator_quick.ipynb     # Interactive inference and ROUGE metric evaluation
```

---

## 🧪 Model Performance & Benchmarking

The models are evaluated on the Extreme Summarization (XSum) test split, where the goal is to produce a single-sentence abstractive summary answering "What is this article about?":

| Model | Parameters | Beam Search Width | Primary Metric (ROUGE-L) | Target Latency |
| :--- | :---: | :---: | :---: | :---: |
| **DistilBART-XSum-12-3** | ~140M | 4 | ~31.2 | Fast (< 1.5s CPU / < 200ms GPU) |
| **BART-Large-XSum** | ~406M | 4–6 | ~37.1 | Balanced |
| **T5-Small (Fine-tuned)** | ~60M | 4 | ~28.4 | Ultra-lightweight Edge/Local |
| **Google PEGASUS-XSum** | ~568M | 8 | ~39.2 | High Precision |

---

## 👤 Author Information

- **Name**: Manoor Sufiyan Ansari
- **Scholar Number**: 24U021005
- **Institution**: Indian Institute of Information Technology Bhopal (IIIT Bhopal)
- **GitHub**: [@Manoor008](https://github.com/Manoor008)
- **Email**: [ansarimanoor786@gmail.com](mailto:ansarimanoor786@gmail.com)
