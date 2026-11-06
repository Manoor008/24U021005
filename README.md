# Automatic News Headline Generator & AIML Repository

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Scholar No**: `24U021005` &nbsp;|&nbsp; **Student**: Manoor Sufiyan Ansari  
**Institute**: Indian Institute of Information Technology Bhopal (IIIT Bhopal)  
**Department**: Department of Computer Science & Engineering / Information Technology  

</div>

---

An end-to-end Natural Language Processing (NLP) and Deep Learning system for abstractive news headline generation using Sequence-to-Sequence (Seq2Seq) Transformer models (Google T5, DistilBART, BART Large, and PEGASUS) fine-tuned on the BBC XSum dataset. The repository includes an interactive Flask web application, complete training and evaluation pipelines in JupyterLab, and the comprehensive AIML laboratory coursework covering unsupervised clustering and heuristic path planning algorithms.

---

## 🏛️ System Architecture

- **Core Model Architectures**:
  - `T5 (Text-to-Text Transfer Transformer)`: Fine-tuned seq2seq headline generation with `"summarize: "` task prefix.
  - `DistilBART-XSum-12-3`: Distilled high-speed abstractive summarization.
  - `BART-Large-XSum` & `Google PEGASUS-XSum`: Deep multi-layer bidirectional encoder and autoregressive decoder.
- **Decoding Mechanism**: Configurable Beam Search (`num_beams` 1–8), length normalization, and n-gram repetition penalties.
- **Backend Framework**: Python, Flask 3.0+, PyTorch (CUDA / CPU), Hugging Face `transformers` and `accelerate`.
- **User Interface**: Integrated lightweight, responsive HTML5/CSS3 interface.
- **Evaluation Framework**: ROUGE metrics (`ROUGE-1`, `ROUGE-2`, `ROUGE-L`) calculated via Hugging Face `evaluate` and `datasets`.
- **Laboratory Modules**: K-Means, Agglomerative Hierarchical, DBSCAN clustering, and A* / Dijkstra path planning.

---

## 📋 Prerequisites

Ensure you have the following installed on your machine:
- **Python** (v3.10 or newer)
- **pip** (Python package manager)
- **PyTorch** (configured with CUDA for GPU acceleration, or CPU)
- **Git**

---

## ⚙️ Environment Configuration

You can configure optional environment variables in `news-headline-generator/.env` or export them directly in your shell:

```env
# Optional: Select a specific model or local checkpoint folder
# Default fallback: sshleifer/distilbart-xsum-12-3
MODEL_NAME=sshleifer/distilbart-xsum-12-3

# Optional: Server port (defaults to 5000)
PORT=5000
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/ansarimanoor786/24U021005.git
cd 24U021005
```

---

### 2. Set Up the Python Environment

```bash
cd news-headline-generator

# (Optional but recommended) Create and activate a virtual environment
python -m venv venv

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 3. Launch the News Headline Generator Web App

```bash
python app.py
```

- **Interactive Web App**: [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **System Health & Device Status**: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)
- **API Endpoint**: `POST http://127.0.0.1:5000/generate`

#### Serving your own fine-tuned model:
If you trained a custom checkpoint (e.g. `./t5-headline-final`) using the provided notebooks:
- **Windows (CMD)**: `set MODEL_NAME=./t5-headline-final && python app.py`
- **Windows (PowerShell)**: `$env:MODEL_NAME="./t5-headline-final"; python app.py`
- **Linux/macOS**: `MODEL_NAME=./t5-headline-final python app.py`

---

### 4. Running Training & Evaluation Notebooks

To fine-tune models or run benchmark evaluations:

```bash
# From within news-headline-generator:
jupyter lab notebooks/
```

- **`t5_xsum_headline_jupyterlab.ipynb`**: Complete end-to-end training pipeline of `t5-small` on the BBC XSum dataset with Hugging Face `Seq2SeqTrainer`.
- **`headline_generator_quick.ipynb`**: Rapid inference test bench and ROUGE scoring against reference ground-truth headlines.

---

## 📁 Project Structure

```text
24U021005/
├── .gitignore                                 # Git ignore rules for virtual environments & model weights
├── README.md                                  # Repository overview and quick start guide
│
├── news-headline-generator/                   # Core Web Application & NLP Pipeline
│   ├── app.py                                 # Flask application, REST API & embedded UI
│   ├── requirements.txt                       # Python dependencies (Flask, Transformers, Torch, etc.)
│   ├── .env.example                           # Example environment variable configuration
│   ├── README.md                              # Application documentation & API specifications
│   └── notebooks/                             # Jupyter notebooks for model training & evaluation
│       ├── t5_xsum_headline_jupyterlab.ipynb  # T5-small fine-tuning pipeline on XSum dataset
│       └── headline_generator_quick.ipynb     # Interactive inference and ROUGE metric evaluation
│
└── lab-assignments/                           # AIML Laboratory Coursework
    ├── README.md                              # Lab documentation, algorithm descriptions & run guide
    └── AIML_Advanced_Clustering_and_Path_Planning.ipynb # Campus service segmentation & A* search
```

---

## 🧪 Laboratory Modules Summary

Inside the [`lab-assignments/`](lab-assignments/) directory:

1. **Campus Service Usage Segmentation**:
   - Advanced unsupervised clustering applied to higher-education service access patterns.
   - Comparative evaluation across **K-Means**, **Agglomerative Hierarchical (Ward Linkage)**, and **DBSCAN**.
   - Hyperparameter selection validated via **Elbow Curvature** (Inertia) and **Silhouette Coefficients**.
   - Latent representation extraction using **Principal Component Analysis (PCA)**.

2. **Heuristic Path Planning & Search**:
   - Optimal path computation across 2D spatial cost grids with obstacles.
   - Comprehensive comparative analysis of **A* Search** (with Manhattan/Euclidean admissible heuristics) versus **Dijkstra's Algorithm** and **Greedy Best-First Search**.
   - Evaluation metrics: Path length optimality, execution latency, and cumulative nodes expanded.

---

## 👤 Author Information

- **Name**: Manoor Sufiyan Ansari
- **Scholar Number**: 24U021005
- **Institution**: Indian Institute of Information Technology Bhopal (IIIT Bhopal)
- **GitHub**: [@ansarimanoor786](https://github.com/ansarimanoor786)
- **Email**: [ansarimanoor786@gmail.com](mailto:ansarimanoor786@gmail.com)

---

## 📄 License

This repository is maintained for academic and educational evaluation at IIIT Bhopal under the MIT License.
