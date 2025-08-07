# ⚖️ CLATBot – AI-Powered Legal Exam Chatbot (Offline)

**CLATBot** is a fully offline, AI-driven chatbot designed to assist students with legal entrance exams like **CLAT(Common Law Admission Test)**.  
It uses **Mistral-7B-Instruct (GGUF format)** in a hybrid RAG (Retrieval-Augmented Generation) setup to deliver contextual, accurate answers from curated and web-scraped legal content.

> The chatbot runs entirely locally — no API keys, no internet required after setup.

---

## ✨ Key Features

- 🧠 **Local LLM (Mistral-7B-Instruct)**: Loaded via `llama-cpp-python` for fully offline inference.
- 🔍 **Semantic Retrieval**: FAISS + Sentence-Transformers for nearest neighbor matching.
- 🌐 **Web Content Integration**: Scraped legal content using `BeautifulSoup`.
- 🌀 **Paraphrasing Module**: Used `Parrot` to generate multiple query styles.
- 💬 **Hybrid RAG Pipeline**: Contextual chunk retrieval + LLM-based answer generation.
- 💻 **Flask Web App**: Interactive frontend built using Flask and Jinja2.

---

## 📂 Project Structure

```
CLATBot/
├── app.py
├── run_clatbot.bat
├── clatbot.ipynb
├── clat_qa_dataset.csv
├── clat_qa_dataset_with_paraphrases.csv
├── faq_web_data.csv
├── faq_web_chunks.csv
├── mistral-7b-instruct-v0.2.Q5_K_M.gguf
├── requirements.txt
├── README.md
├── static/
├── templates/
└── text-generation-webui/
```

---

## 📦 Dependencies

Listed in `requirements.txt`:

```
Flask
pandas
torch
sentence-transformers
beautifulsoup4
requests
parrot
llama-cpp-python
scikit-learn
jinja2
numpy
gunicorn
faiss-cpu
```

> 🐍 Python 3.10+ recommended.

---

## 🔧 Installation

### Step 1: Clone the Repo

```bash
git clone https://github.com/yourusername/CLATBot.git
cd CLATBot
```

### Step 2: (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Mistral Model (GGUF)

Model Link: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3  
Download the `.gguf` file manually or use:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="models/mistral",
    allow_patterns=["*.gguf"]
)
```

Move the `.gguf` file to the root directory.

---

## ▶️ Running the App

```bash
python app.py
```

Then open: `http://127.0.0.1:5000/`

Alternatively on Windows:

```bash
run_clatbot.bat
```

---

## 🧠 Architecture (How it works)

1. User enters a legal query on the website.
2. Query is encoded using Sentence-Transformers.
3. FAISS finds top-k most similar chunks from FAQ corpus.
4. Selected content is sent to Mistral LLM via `llama-cpp-python`.
5. LLM generates a context-aware response.
6. Result is rendered on the frontend.

---

## 📊 Datasets Used

| File Name                              | Purpose                                  |
|----------------------------------------|------------------------------------------|
| `clat_qa_dataset.csv`                  | Core legal Q&A                           |
| `clat_qa_dataset_with_paraphrases.csv`| Enhanced Q&A with paraphrasing (Parrot)  |
| `faq_web_data.csv`                     | Web-scraped raw legal content            |
| `faq_web_chunks.csv`                   | Pre-chunked version for FAISS retrieval  |

---

## 🧪 Technologies Used

- **Languages**: Python, HTML, CSS
- **Libraries**: `Flask`, `FAISS`, `Sentence-Transformers`, `BeautifulSoup`, `Parrot`, `Pandas`, `NumPy`, `Torch`, `scikit-learn`, `llama-cpp-python`, `Gunicorn`, `Jinja2`
- **Model**: [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

---

## 📜 License

Apache License 2.0  
Model credit: [Mistralai / Mistral-7B-Instruct](https://huggingface.co/mistralai)
