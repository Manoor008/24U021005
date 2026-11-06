# Automatic News Headline Generator (Abstractive Summarization)

An end-to-end NLP web application and fine-tuning pipeline for generating succinct, coherent, and click-worthy news headlines from full-length articles using abstractive text summarization.

---

## 🚀 Key Features

- **Multi-Model Support**: Pre-configured support for cutting-edge Seq2Seq architectures:
  - `sshleifer/distilbart-xsum-12-3` (Fast, lightweight inference)
  - `facebook/bart-large-xsum` (High quality & semantic accuracy)
  - `google/pegasus-xsum` (Specialized summarization pretraining)
  - Fine-tuned `t5-small` / `t5-base` checkpoints
- **Interactive Web Interface**: Clean, responsive frontend served directly with Flask.
- **RESTful API**: `/generate` and `/health` endpoints for easy programmatic integration.
- **Configurable Beam Search**: Supports greedy search and beam search decoding with configurable width (`num_beams`), length limits, and n-gram repetition blocking.
- **Complete Training Pipeline**: Interactive Jupyter notebooks for fine-tuning Google's T5 architecture on the BBC XSum dataset with Hugging Face `transformers` and `evaluate`.

---

## 📋 Prerequisites

- **Python**: v3.10 or newer
- **PyTorch**: Recommended with CUDA GPU support for faster generation and training

---

## 🛠️ Installation & Setup

1. **Navigate to the directory**:
   ```bash
   cd news-headline-generator
   ```

2. **(Optional) Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows (cmd/PowerShell):
   .\venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Web Application

To run using the default pre-trained model:
```bash
python app.py
```

The application will start at:
- **Web Interface**: [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Health Check**: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

### Serving a Custom Fine-Tuned Model:
If you have trained your own checkpoint using the included notebooks:
- **Windows (CMD)**:
  ```cmd
  set MODEL_NAME=./t5-headline-final && python app.py
  ```
- **Windows (PowerShell)**:
  ```powershell
  $env:MODEL_NAME="./t5-headline-final"; python app.py
  ```
- **macOS / Linux**:
  ```bash
  MODEL_NAME=./t5-headline-final python app.py
  ```

---

## 📡 API Endpoints

### 1. `POST /generate`
Generate a headline from an article text.

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
Returns system status, active hardware acceleration device (`cuda` or `cpu`), and loaded model identifier.

---

## 📓 Jupyter Notebooks

Located inside the [`notebooks/`](notebooks/) directory:

1. **`t5_xsum_headline_jupyterlab.ipynb`**:
   - Complete production fine-tuning notebook for `t5-small` on the BBC XSum dataset.
   - Includes dataset tokenization, `DataCollatorForSeq2Seq`, `Seq2SeqTrainer`, ROUGE evaluation metric computation, and model checkpoint export.
2. **`headline_generator_quick.ipynb`**:
   - Rapid evaluation and inference testing pipeline.
   - Compares zero-shot vs fine-tuned models side-by-side with reference XSum summaries.
