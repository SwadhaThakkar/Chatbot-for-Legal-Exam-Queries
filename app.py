from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import faiss
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Clean text utility
def clean_text(text):
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).strip()

# Load Data
df_expanded = pd.read_csv("clat_qa_dataset_with_paraphrases.csv")
df_chunks = pd.read_csv("faq_web_chunks.csv")

df_expanded["clean_question"] = df_expanded["question"].apply(clean_text)
df_chunks["clean_chunk"] = df_chunks["chunk"].apply(clean_text)

faq_corpus = df_expanded["clean_question"].tolist()
web_corpus = df_chunks["clean_chunk"].tolist()

# Embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
faq_embeddings = model.encode(faq_corpus, convert_to_tensor=True)
web_embeddings = model.encode(web_corpus, convert_to_tensor=True)

# FAISS for fallback
web_index = faiss.IndexFlatL2(web_embeddings.shape[1])
web_index.add(web_embeddings.cpu().numpy())

# Load LLM
llm = Llama(
    model_path="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    n_ctx=4096,
    n_threads=8
)

# Friendly Responses
gratitude_keywords = {"thanks", "thank", "thankyou", "ty"}
positive_keywords = {"good", "nice", "fine", "okay", "ok", "alright", "welcome"}
greeting_keywords = {"hi", "hello", "hey", "good morning", "good evening"}

# Hybrid Answer Generator
def generate_answer(user_query, top_k=3):
    cleaned = clean_text(user_query)
    words = set(cleaned.split())

    if words & gratitude_keywords:
        return "You're welcome! 😊 Let me know if you have more CLAT questions."
    if words & greeting_keywords:
        return "Hello! 👋 How can I assist you with your CLAT queries today?"
    if words & positive_keywords:
        return "Glad to hear that! ✅ What else would you like to know?"

    # MiniLM attempt
    query_embedding = model.encode([cleaned], convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)
    best_idx = torch.argmax(scores).item()
    if scores[0][best_idx] >= 0.65:
        return df_expanded.iloc[best_idx]["answer"]

    # Fallback: Mistral
    D, I = web_index.search(query_embedding.cpu().numpy(), top_k)
    context = "\n".join(df_chunks.iloc[i]["chunk"] for i in I[0])
    prompt = f"""Use the following context to answer the question.

### Context:
{context}

### Question:
{user_query}

### Answer:"""
    try:
        output = llm(prompt, max_tokens=300)
        return output["choices"][0]["text"].strip()
    except Exception as e:
        logger.error("LLM error", exc_info=e)
        return "⚠️ Sorry, I couldn't generate an answer right now."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    try:
        user_query = request.get_json().get("message", "")
        if not user_query.strip():
            return jsonify({"reply": "❓ Please enter a valid question."})
        reply = generate_answer(user_query)
        return jsonify({"reply": reply})
    except Exception as e:
        logger.exception("Chat error")
        return jsonify({"reply": "⚠️ An unexpected error occurred."})

if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# from llama_cpp import Llama
# import pandas as pd
# import torch
# import re
# from sentence_transformers import SentenceTransformer
# import faiss
# import logging

# # ----------------------------
# # 🔧 Logging Configuration
# # ----------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ----------------------------
# # 🔧 Flask App Init
# # ----------------------------
# app = Flask(__name__)

# # ----------------------------
# # 🧠 Load LLM
# # ----------------------------
# llm = Llama(
#     model_path="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
#     n_ctx=4096,
#     n_threads=8
# )

# # ----------------------------
# # 🧹 Text Cleaning
# # ----------------------------
# def clean(text):
#     text = str(text).lower()
#     return re.sub(r"[^a-z0-9\s]", "", text).strip()

# # ----------------------------
# # 📄 Load and Preprocess Data
# # ----------------------------
# datasets = {
#     "faq_main": pd.read_csv("clat_qa_dataset.csv"),
#     "faq_paraphrased": pd.read_csv("clat_qa_dataset_with_paraphrases.csv"),
#     "web_data": pd.read_csv("faq_web_data.csv"),
#     "web_chunks": pd.read_csv("faq_web_chunks.csv"),
# }

# for key in datasets:
#     df = datasets[key]
#     if "question" in df.columns:
#         df["clean"] = df["question"].apply(clean)
#     elif "chunk" in df.columns:
#         df["clean"] = df["chunk"].apply(clean)
#     elif "text" in df.columns:
#         df["clean"] = df["text"].apply(clean)

# # Combine all cleaned text for embedding
# all_texts = []
# text_sources = []

# for key, df in datasets.items():
#     for i, row in df.iterrows():
#         all_texts.append(row["clean"])
#         text_sources.append((key, i))

# # ----------------------------
# # 🤖 Embedding with SentenceTransformer
# # ----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
# embeddings = embedder.encode(all_texts, convert_to_tensor=False)

# # ----------------------------
# # ⚡ FAISS Index
# # ----------------------------
# dim = embeddings[0].shape[0]
# index = faiss.IndexFlatL2(dim)
# index.add(embeddings)

# # ----------------------------
# # 🙋 Friendly Keywords
# # ----------------------------
# gratitude_keywords = {"thanks", "thank", "thankyou", "ty"}
# positive_keywords = {"good", "nice", "fine", "okay", "ok", "alright", "welcome"}
# greeting_keywords = {"hi", "hello", "hey", "good morning", "good evening"}

# # ----------------------------
# # 🔎 RAG Answer Generator
# # ----------------------------
# def generate_answer(query, top_k=3):
#     cleaned_query = clean(query)

#     # Friendly shortcuts
#     words = set(cleaned_query.split())
#     if words & gratitude_keywords:
#         return "You're welcome! 😊"
#     if words & greeting_keywords:
#         return "Hello! 👋 How can I help you with CLAT today?"
#     if words & positive_keywords:
#         return "Glad to hear that! ✅ What would you like to know?"

#     # Embed and search
#     query_embedding = embedder.encode([cleaned_query])
#     D, I = index.search(query_embedding, top_k)

#     if D[0][0] > 0.5:
#         # Not similar enough
#         return "Sorry, I couldn't find enough information on that."

#     # Prepare context from matched results
#     context_parts = []
#     for idx in I[0]:
#         source_key, source_row = text_sources[idx]
#         source_df = datasets[source_key]
#         if "answer" in source_df.columns:
#             context_parts.append(source_df.iloc[source_row]["answer"])
#         elif "chunk" in source_df.columns:
#             context_parts.append(source_df.iloc[source_row]["chunk"])
#         elif "text" in source_df.columns:
#             context_parts.append(source_df.iloc[source_row]["text"])

#     context = "\n".join(context_parts)
#     prompt = f"""Use the following context to answer the question.

# ### Context:
# {context}

# ### Question:
# {query}

# ### Answer:"""

#     try:
#         response = llm(prompt, max_tokens=300)
#         answer = response["choices"][0]["text"].strip()
#         return answer
#     except Exception as e:
#         logger.error(f"LLM error: {e}")
#         return "⚠️ Sorry, something went wrong while generating the answer."

# # ----------------------------
# # 🌐 Flask Routes
# # ----------------------------
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get", methods=["POST"])
# def get_bot_response():
#     try:
#         data = request.get_json()
#         user_query = data.get("message", "")
#         if not user_query:
#             return jsonify({"reply": "❓ Please enter a valid question."})

#         reply = generate_answer(user_query)
#         return jsonify({"reply": reply})
#     except Exception as e:
#         logger.exception("Error during chat response")
#         return jsonify({"reply": "⚠️ Sorry, something went wrong."})

# # ----------------------------
# # 🚀 Run Server
# # ----------------------------
# if __name__ == "__main__":
#     app.run(debug=True)
