# 🎙️ Sourashtra Smart Voice Phrasebook & Translator

A dual-framework (Streamlit & Gradio) Proof-of-Concept (PoC) application designed for a 4-day phrasebook collection and real-time LLM in-context few-shot translation of the Sourashtra language into English and Tamil.

---

## 🌟 Features

- **🎙️ Voice Collector (Father's UI)**: 1-tap mobile audio recording and transliteration collection across 30 daily English prompts.
- **🚀 Live Translator (xAI Grok / Gemini / OpenAI)**: Dynamically constructs few-shot prompt context from `metadata.csv` and outputs strict JSON translations `{"english": "...", "tamil": "..."}`.
- **🔊 Text-to-Speech Output**: Generates instant audio playback in English and Tamil via `gTTS`.
- **📊 Dataset Explorer**: Interactive inspection, metric tracking, and CSV dataset export.
- **⚡ Dual Framework**: Run via **Streamlit** or **Gradio**.

---

## 📁 Repository Structure

```text
sourashtra_app/
├── streamlit_app.py            # Streamlit Application (Primary Cloud Entrypoint)
├── app.py                      # Gradio Application (Gradio Web UI)
├── requirements.txt            # Project Dependencies
├── prompts.txt                 # Pre-populated 30 Daily English Prompts
├── .gitignore                  # Git Ignore Rules
├── README.md                   # Setup & Deployment Guide
├── .streamlit/
│   └── config.toml             # Streamlit Theme Configuration
└── sourashtra_dataset/
    ├── metadata.csv            # Structured Dataset (id, timestamp, english_phrase, sourashtra_text, audio_file)
    └── audio/                  # Saved Audio Recordings (.wav)
```

---

## 🚀 Quickstart (Local Execution)

### 1. Clone & Install Dependencies

```bash
cd sourashtra_app
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional)

Set one of the following API keys for live LLM inference:

```bash
# Windows PowerShell
$env:XAI_API_KEY="your-grok-api-key"
# or
$env:GEMINI_API_KEY="your-gemini-api-key"
# or
$env:OPENAI_API_KEY="your-openai-api-key"
```

> *Note: If no API key is provided, the application runs in Offline Dataset Fallback Mode.*

### 3. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

### 4. Run Gradio App

```bash
python app.py
```

---

## 🐙 Step-by-Step GitHub Publishing Guide

### 1. Initialize Git Repository

In your local terminal inside `sourashtra_app/`:

```bash
git init
git add .
git commit -m "Initial commit of Sourashtra Smart Voice Translator"
```

### 2. Push to GitHub

1. Create a new public or private repository on [GitHub](https://github.com/new) named `sourashtra-translator`.
2. Connect and push your local commits:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/sourashtra-translator.git
git push -u origin main
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app** -> **Use existing repo**.
3. Select your repository: `YOUR_GITHUB_USERNAME/sourashtra-translator`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `streamlit_app.py`.
6. Expand **Advanced Settings** -> **Secrets** and add your API keys:

```toml
XAI_API_KEY = "your-xai-grok-key"
GEMINI_API_KEY = "your-gemini-api-key"
OPENAI_API_KEY = "your-openai-api-key"
```

7. Click **Deploy!** Your app will be live on a public URL! 🎉

---

## 📜 License

MIT License. Designed for preserve and translate minority language speech data.
