# Vamsi AI Assistant 🤖

An AI-powered chatbot built using **Python, Streamlit, and Ollama**. It runs large language models locally on your computer and supports **PDF-based Retrieval-Augmented Generation (RAG)**, allowing users to ask questions about uploaded PDF documents.

## 🚀 Project Overview

Vamsi AI Assistant provides a ChatGPT-like conversational interface powered by locally running open-source language models. It also enables users to upload PDF documents and ask questions based on their content.

The project is designed to run locally without requiring OpenAI API credits.

## ✨ Features

* 🤖 Interactive AI chatbot interface
* 🧠 Local LLM integration using Ollama
* 📄 PDF document upload and text extraction
* 🔍 Retrieval-Augmented Generation (RAG) for document-based question answering
* 🧩 Text chunking and embedding generation
* 📊 Semantic similarity-based document retrieval
* 💬 Conversational chat interface using Streamlit
* 🔒 Local model execution without requiring a cloud LLM API

## 🛠️ Technologies Used

* **Python** — Application logic
* **Streamlit** — User interface
* **Ollama** — Local LLM inference
* **Llama 3.2** — Chat model
* **nomic-embed-text** — Text embeddings
* **PyPDF2** — PDF text extraction
* **NumPy** — Vector operations and similarity calculations
* **Requests** — Communication with the Ollama API

## 🏗️ Architecture

```text
User
  |
  v
Streamlit Chat Interface
  |
  v
Vamsi AI Assistant
  |
  +----------------------+
  |                      |
  v                      v
Ollama Chat API      PDF Processing
  |                      |
  v                      v
Local LLM            Text Chunking
                         |
                         v
                    Text Embeddings
                         |
                         v
                    Similarity Search
                         |
                         v
                  Relevant PDF Context
                         |
                         v
                   LLM Response
```

## ⚙️ Prerequisites

Before running the project, install:

* Python 3.10 or later
* Git (if you want to clone the repository)
* Ollama
* A compatible local chat model and embedding model

## 📥 Installation and Setup

### Step 1 — Install Ollama

Download and install Ollama from:

https://ollama.com

Verify the installation:

```bash
ollama --version
```

### Step 2 — Download the required models

Pull the chat model:

```bash
ollama pull llama3.2
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Make sure the model names match those configured in the application.

### Step 3 — Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/vamsi-ai-assistant.git
cd vamsi-ai-assistant
```

Replace `YOUR-USERNAME` with your GitHub username.

If you already have the project on your computer, open its folder in VS Code instead.

### Step 4 — Create a virtual environment

```bash
python -m venv venv
```

### Step 5 — Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### Step 6 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 7 — Start Ollama

Ollama usually starts automatically after installation. If it is not running, start it using:

```bash
ollama serve
```

Keep the Ollama service running while using the chatbot.

### Step 8 — Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in your terminal. Open that URL in your browser to use Vamsi AI Assistant.

## 📄 How to Use

1. Start Ollama and run the Streamlit application.
2. Open the chatbot in your browser.
3. Enter a message to chat with the AI assistant.
4. Upload a PDF document using the document upload option.
5. Allow the application to extract and process the PDF text.
6. Ask questions about the uploaded document.
7. The application retrieves relevant text and uses it to help generate an answer.

Simple greetings can be handled as normal conversations without retrieving PDF content.

## 🔐 Privacy and Security

* Chat and embedding models run locally through Ollama.
* PDF processing is performed by the application on your computer.
* Avoid uploading confidential documents to public repositories.
* Never commit passwords, API keys, private documents, or other sensitive information to GitHub.

**Note:** Local model execution does not automatically guarantee that every part of an application is private. Review any external network requests and deployment configuration before using sensitive data.

## 🔮 Future Improvements

* Support for multiple PDF documents
* Improved document retrieval and response accuracy
* Chat history management
* Additional local language models
* Improved error handling and user experience
* Deployment options for remote access

## 👨‍💻 Author

**Vamsi Ande**

AI/ML Graduate | Python Developer | AI Enthusiast

## 📜 License

This project is available for learning and development. Add a license file if you decide to distribute it under a specific open-source license.
