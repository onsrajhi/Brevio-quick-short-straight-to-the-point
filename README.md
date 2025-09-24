# Brevio-quick-short-straight-to-the-point

Brevio is a Streamlit-based summarization app that allows users to generate intelligent summaries with audio! (both formats downloadable) from:

1. 🎥 YouTube videos
2. 🌐 Web URLs
3. 📄 PDF documents

Powered by **LangChain**, **Groq LLM (Gemma2-9b-it)**, and **gTTS**, Zummary makes digesting lengthy content easier, faster, and accessible.

---

## ✨ Features

**🔍 Detailed Summaries:** Introduction, key points, and conclusion with a suitable title.

**🎧 Audio Output:** Listen to MP3 of the summary using gTTS.

**📥 Text & Audio Downloads:** Get both text and audio formats of the summary.

**🤖 LLM-Driven:** Uses LangChain and Groq's Gemma model for high-quality summarization.

**🧠 Smart Mode Switching:** Automatically selects summarization method (stuff vs map_reduce) based on content length.

**✅ Supports:**

1. YouTube video transcripts
2. Unstructured website content
3. PDF documents

---

## 🧩 Tech Stack

**Streamlit** — UI & interaction

**LangChain** — LLM integration and summarization chains

**ChatGroq** — Gemma2-9b-it model

**gTTS** — Text-to-speech for summary audio

**YouTubeTranscriptAPI** — Fetching video transcripts

**UnstructuredURLLoader, PyPDFLoader** — Web & PDF parsing

**PIL, Base64, Regex** — Utility handling for media and content

---

## 🚀 Getting Started

1. Clone the Repo

> git clone https://github.com/Adityajain8595/Zummary-Content-Summarizer-App.git

> cd Zummary-Content-Summarizer-App

2. Install Dependencies

It’s recommended to use a virtual environment.

> pip install -r requirements.txt

3. Environment Setup

Create a .env file in the root directory:

> LANGCHAIN_API_KEY=your_langchain_key

> GROQ_API_KEY=your_groq_key

(Only needed if you're using proxies)

> proxy_username=your_proxy_username   

> proxy_password=your_proxy_password

Add these secrets to .streamlit/secrets.toml as well (used in app):

> LANGCHAIN_API_KEY = "your_langchain_key"

> GROQ_API_KEY = "your_groq_key"

4. Run the App

> streamlit run summarize.py

---

## 📸 Screenshots

https://github.com/user-attachments/assets/bd4deeba-f959-4587-a515-2f9077b47f8a

---

## 👨‍💻 Author

Aditya Jain

Would love to connect with you and hear your feedback! 

Connect with me on 📧 LinkedIn: https://www.linkedin.com/in/adityajain8595/


