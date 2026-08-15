# 🛡️ Defence AI Assistant

An AI-powered question-answering system for a Defence/DRDO knowledge base using Retrieval-Augmented Generation (RAG).

## 🚀 Project Overview

Defence AI Assistant is an AI-based application that allows users to ask questions about information available in the provided Defence/DRDO dataset.

The system retrieves relevant information from the knowledge base using vector similarity search and then uses an AI model to generate a concise answer based on the retrieved information.

The project is available as both a web application and an Android application.

## ✨ Features

- 🔎 Semantic search over Defence/DRDO documents
- 🤖 AI-powered question answering
- 📚 Retrieval-Augmented Generation (RAG)
- 🧠 HuggingFace embeddings
- 🗄️ ChromaDB vector database
- 🌐 Streamlit web application
- 📱 Android application
- 📄 Source information displayed with answers
- 🛡️ Answers restricted to the provided knowledge base

## 🏗️ System Architecture

User Question
      ↓
HuggingFace Embeddings
      ↓
ChromaDB Vector Search
      ↓
Relevant Defence Documents
      ↓
Context Creation
      ↓
Gemini AI
      ↓
Generated Answer
      ↓
User

## 🛠️ Technologies Used

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Google Gemini API
- Streamlit
- Android Studio
- Git & GitHub

## 📂 Project Structure

```text
DRDO/
│
├── app.py
├── query.py
├── ingest.py
├── download_documents.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│
└── android/
