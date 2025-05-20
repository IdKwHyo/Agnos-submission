# Agnos-submission
# 🤖 Agnos Health Forum Chatbot – Candidate Submission

This project is a submission for the Agnos Health candidate assignment. It is an **LLM-powered chatbot** designed to answer health-related questions by retrieving relevant information from the [Agnos Health Forum](https://www.agnoshealth.com/forums). It uses a **RAG (Retrieval-Augmented Generation)** pipeline with local embeddings, scraping, and a chat interface.

---

## 🧠 Features

- 🔍 **Web Scraper**: Automatically extracts forum posts
- 🧾 **Vector Store (Chroma)**: Stores and indexes forum data locally
- 🤖 **LLM Retrieval Chain**: Uses HuggingFace models to answer queries
- 💬 **Chat UI**: Clean Flask frontend for user interaction
- 🔄 **Live Scrape Button**: Refresh data anytime from the interface
- 🔒 **Fully Local Execution**: No OpenAI/GPT dependencies
- 🧠 **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

---

## 📂 Repository Structure

Agnos-submission/
│
├── agnos1.py # Main backend app with Flask and LangChain
├── forum_data.csv # Scraped data file (generated on first run)
├── templates/
│ └── index.html # Chat UI
├── static/ # (Optional static files)
├── db/ # Local vector DB (generated)
├── requirements.txt # Python dependencies
└── README.md # You are here

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.10+
- Hugging Face API Token (only required for HF-hosted models)

### 🔌 Install Dependencies

```bash
git clone https://github.com/IdKwHyo/Agnos-submission.git
cd Agnos-submission
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python agnos1_3.py

make sure to have. env containing all api key needed.
