# 🧠 CLAT Chatbot: Legal Exam FAQ Assistant

This project builds a chatbot that answers questions about the **Common Law Admission Test (CLAT)** using an NLP-powered approach. It uses a combination of a curated FAQ dataset, paraphrased questions, and web-scraped content for fallback responses.

---

## 💡 Problem Statement

Design a chatbot to answer CLAT-related queries by matching user questions with an expanded FAQ dataset and web-based information using semantic similarity.

---

## 📁 Project Structure

- `clat_chatbot_faq.ipynb` : Main notebook with data loading, web scraping, embedding generation, and chatbot logic.
- `clat_qa_dataset.csv` : Core dataset with question-answer pairs.
- `clat_qa_dataset_with_paraphrases.csv` : Expanded dataset with paraphrased versions of questions.
- `faq_web_data.csv` : Web-scraped FAQ content from trusted CLAT resources.
- `requirements.txt` : List of Python libraries required to run the notebook.
- `README.md` : Project overview, setup instructions, and usage.

---

## ⚙️ Setup Instructions

1. Clone the repository or unzip the folder.
2. Install dependencies (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
