---
title: Indian Legal Assistant BNS 2023
emoji: ⚖️
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: true
---

# ?? Indian Legal Assistant - BNS 2023

A RAG-powered chatbot for the Bharatiya Nyaya Sanhita (BNS) 2023.

## What It Does
- Answers grounded in the official BNS 2023 document (237 pages)
- Cites exact section numbers and page numbers
- Multi-turn follow-up questions via conversation memory
- Hybrid retrieval: BM25 + semantic search via RRF
- Refuses to hallucinate - says so if answer not in document

## Tech Stack
- LLM: Llama 3.3 70B via Groq API
- Embeddings: all-MiniLM-L6-v2
- Vector DB: ChromaDB
- Keyword Search: BM25
- Orchestration: LangChain LCEL + memory
- UI: Gradio ChatInterface

## Disclaimer
For educational purposes only. Not legal advice.

---
Developed by Faisal Imam
GitHub: https://github.com/faisalimam1
LinkedIn: https://www.linkedin.com/in/faisalimam19
HuggingFace: https://huggingface.co/faisalimam19
