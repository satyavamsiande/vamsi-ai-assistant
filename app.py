# ============================================================
# VAMSI AI ASSISTANT
# Features:
# - Local LLM with Ollama
# - Chat Memory
# - Streaming
# - Temperature Control
# - PDF Upload
# - RAG / Document Question Answering
# - Semantic Search
# - Custom Dancing Robot UI
# - Normal Chat + PDF Chat
# ============================================================

import json
import requests
import streamlit as st
import numpy as np
import PyPDF2


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vamsi AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

CHAT_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# CUSTOM CSS + DANCING ROBOT
# ============================================================

st.html("""
<style>

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}


/* ============================================================
   HEADER
   ============================================================ */

.hero-card {
    background: linear-gradient(135deg, #111827, #020617);
    border: 1px solid rgba(59,130,246,0.40);
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 22px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
}

.robot-area {
    display: flex;
    align-items: center;
    gap: 24px;
}


/* ============================================================
   ROBOT
   ============================================================ */

.robot {
    width: 85px;
    height: 110px;
    position: relative;
    animation: robotDance 1.2s infinite ease-in-out;
    transform-origin: bottom center;
    flex-shrink: 0;
}

.robot-head {
    width: 58px;
    height: 48px;
    background: linear-gradient(145deg, #e5e7eb, #64748b);
    border-radius: 15px;
    position: absolute;
    top: 0;
    left: 14px;
    border: 3px solid #38bdf8;
    box-shadow: 0 0 12px #38bdf8;
}

.robot-eye {
    width: 9px;
    height: 9px;
    background: #22d3ee;
    border-radius: 50%;
    position: absolute;
    top: 18px;
    box-shadow: 0 0 10px #22d3ee;
}

.robot-eye.left {
    left: 13px;
}

.robot-eye.right {
    right: 13px;
}

.robot-mouth {
    width: 25px;
    height: 5px;
    background: #0f172a;
    border-radius: 10px;
    position: absolute;
    bottom: 10px;
    left: 14px;
}

.robot-body {
    width: 55px;
    height: 48px;
    background: linear-gradient(145deg, #94a3b8, #334155);
    border-radius: 12px;
    position: absolute;
    top: 55px;
    left: 15px;
    border: 3px solid #38bdf8;
    box-shadow: 0 0 12px rgba(56,189,248,0.5);
}

.robot-light {
    width: 13px;
    height: 13px;
    background: #22c55e;
    border-radius: 50%;
    position: absolute;
    top: 15px;
    left: 18px;
    box-shadow: 0 0 12px #22c55e;
    animation: blink 0.8s infinite alternate;
}

.robot-arm {
    width: 12px;
    height: 42px;
    background: linear-gradient(145deg, #cbd5e1, #475569);
    border-radius: 10px;
    position: absolute;
    top: 58px;
}

.robot-arm.left {
    left: 1px;
    transform: rotate(25deg);
    animation: leftArm 0.6s infinite alternate;
}

.robot-arm.right {
    right: 1px;
    transform: rotate(-25deg);
    animation: rightArm 0.6s infinite alternate;
}

.robot-antenna {
    width: 5px;
    height: 13px;
    background: #38bdf8;
    position: absolute;
    top: -13px;
    left: 40px;
}

.robot-antenna-light {
    width: 9px;
    height: 9px;
    background: #f43f5e;
    border-radius: 50%;
    position: absolute;
    top: -20px;
    left: 38px;
    box-shadow: 0 0 10px #f43f5e;
    animation: blink 0.5s infinite alternate;
}


/* ============================================================
   ROBOT ANIMATION
   ============================================================ */

@keyframes robotDance {

    0% {
        transform: rotate(-7deg) translateY(0px);
    }

    25% {
        transform: rotate(7deg) translateY(-8px);
    }

    50% {
        transform: rotate(-7deg) translateY(-14px);
    }

    75% {
        transform: rotate(7deg) translateY(-8px);
    }

    100% {
        transform: rotate(-7deg) translateY(0px);
    }
}

@keyframes leftArm {

    from {
        transform: rotate(25deg);
    }

    to {
        transform: rotate(-35deg);
    }
}

@keyframes rightArm {

    from {
        transform: rotate(-25deg);
    }

    to {
        transform: rotate(35deg);
    }
}

@keyframes blink {

    from {
        opacity: 0.4;
    }

    to {
        opacity: 1;
    }
}


/* ============================================================
   TEXT
   ============================================================ */

.brand-title {
    font-size: 34px;
    font-weight: 800;
    color: #f8fafc;
}

.brand-subtitle {
    font-size: 16px;
    color: #d1d5db;
    margin-top: 5px;
}

.blue {
    color: #38bdf8;
}

.small-note {
    color: #9ca3af;
    font-size: 13px;
    margin-top: 8px;
}


/* ============================================================
   DOCUMENT STATUS
   ============================================================ */

.document-box {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(59,130,246,0.3);
    padding: 12px;
    border-radius: 12px;
    margin-top: 10px;
}

</style>


<div class="hero-card">

    <div class="robot-area">

        <div class="robot">

            <div class="robot-antenna"></div>
            <div class="robot-antenna-light"></div>

            <div class="robot-head">
                <div class="robot-eye left"></div>
                <div class="robot-eye right"></div>
                <div class="robot-mouth"></div>
            </div>

            <div class="robot-arm left"></div>
            <div class="robot-arm right"></div>

            <div class="robot-body">
                <div class="robot-light"></div>
            </div>

        </div>


        <div>

            <div class="brand-title">
                Vamsi AI <span class="blue">Assistant</span>
            </div>

            <div class="brand-subtitle">
                Your personal AI learning assistant powered by Python,
                Streamlit and Ollama.
            </div>

            <div class="small-note">
                Chat with AI, upload PDFs and ask questions using RAG.
            </div>

        </div>

    </div>

</div>
""")


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# ============================================================
# HELPER - CHECK OLLAMA
# ============================================================

def check_ollama_running():

    try:

        response = requests.get(
            "http://localhost:11434",
            timeout=3
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:

        return False


# ============================================================
# HELPER - CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return np.array(data["embeddings"][0])


# ============================================================
# HELPER - SPLIT TEXT
# ============================================================

def split_text(text, chunk_size=800, overlap=150):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# HELPER - EXTRACT PDF TEXT
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# HELPER - PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    text = extract_pdf_text(uploaded_file)

    if not text.strip():
        return []

    chunks = split_text(text)

    documents = []

    progress = st.progress(0)

    for index, chunk in enumerate(chunks):

        embedding = create_embedding(chunk)

        documents.append(
            {
                "text": chunk,
                "embedding": embedding
            }
        )

        progress.progress(
            (index + 1) / len(chunks)
        )

    progress.empty()

    return documents


# ============================================================
# HELPER - COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


# ============================================================
# HELPER - RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=3,
    minimum_score=0.30
):

    if not st.session_state.documents:
        return []

    query_embedding = create_embedding(query)

    scored_documents = []

    for document in st.session_state.documents:

        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        scored_documents.append(
            (
                score,
                document["text"]
            )
        )

    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    relevant_documents = []

    for score, text in scored_documents[:top_k]:

        if score >= minimum_score:

            relevant_documents.append(text)

    return relevant_documents


# ============================================================
# HELPER - CHECK SIMPLE GREETINGS
# ============================================================

def is_greeting(text):

    text = text.lower().strip()

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "morning",
        "afternoon",
        "evening",
        "hey there",
        "hello there",
        "hi there"
    }

    return text in greetings


# ============================================================
# HELPER - CHECK CASUAL QUESTIONS
# ============================================================

def is_casual_message(text):

    text = text.lower().strip()

    casual_messages = {
        "how are you",
        "how are you?",
        "how r u",
        "how r u?",
        "what's up",
        "whats up",
        "what are you doing",
        "who are you",
        "tell me about yourself",
        "thank you",
        "thanks",
        "thankyou",
        "bye",
        "goodbye"
    }

    return text in casual_messages


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ AI Settings")


    # ========================================================
    # MODEL
    # ========================================================

    model = st.selectbox(
        "Choose AI Model",
        [
            "llama3.2",
            "llama3.1",
            "llama3",
            "mistral",
            "gemma2",
            "qwen2.5",
            "qwen2.5:3b"
        ],
        index=0
    )


    # ========================================================
    # TEMPERATURE
    # ========================================================

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = st.text_area(
        "AI Assistant Instructions",

        value="""
You are Vamsi's personal AI learning assistant.

Help the user understand:

- Python
- SQL
- Machine Learning
- Deep Learning
- Generative AI
- LLMs
- RAG
- FastAPI
- PostgreSQL
- Software Development

Explain concepts simply and clearly.

When useful:

- Give examples
- Provide code
- Explain code step by step
- Mention common mistakes
- Give interview-oriented explanations

IMPORTANT BEHAVIOR:

1. Greetings such as:
   Hi
   Hello
   Hey
   Good morning
   Good afternoon
   Good evening
   Good night

   should receive a natural and friendly greeting.

2. Do not use the PDF for simple greetings
   or normal casual conversation.

3. For normal technical questions, answer normally.

4. If relevant PDF context is provided,
   use that context to answer questions about the PDF.

5. If the answer is not available in the PDF context,
   clearly say that the information was not found
   in the uploaded document.

6. Never say that a PDF has not been uploaded
   when a PDF is available.

7. Do not force every question to use the PDF.

8. Never invent information from the PDF.
""",

        height=250
    )


    st.markdown("---")


    # ========================================================
    # PDF UPLOAD
    # ========================================================

    st.markdown("### 📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )


    if uploaded_file is not None:

        if (
            st.session_state.document_name
            != uploaded_file.name
        ):

            st.session_state.documents = []

            st.session_state.document_name = (
                uploaded_file.name
            )

            with st.spinner(
                "Reading and indexing your PDF..."
            ):

                try:

                    st.session_state.documents = (
                        process_pdf(uploaded_file)
                    )

                    if st.session_state.documents:

                        st.success(
                            f"PDF ready: {uploaded_file.name}"
                        )

                    else:

                        st.error(
                            "Could not extract text from this PDF."
                        )

                except Exception as e:

                    st.error(
                        f"PDF processing failed: {e}"
                    )


    # ========================================================
    # DOCUMENT STATUS
    # ========================================================

    if st.session_state.documents:

        st.markdown(
            f"""
            <div class="document-box">
                📄 <b>{st.session_state.document_name}</b><br>
                🧩 {len(st.session_state.documents)} chunks indexed
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


    # ========================================================
    # REMOVE PDF
    # ========================================================

    if st.button("📄 Remove PDF"):

        st.session_state.documents = []

        st.session_state.document_name = None

        st.rerun()


    st.markdown("---")


    # ========================================================
    # PROJECT FEATURES
    # ========================================================

    st.markdown("### 🚀 Project Features")

    st.markdown("""
    - 🤖 Local AI Chatbot
    - 🧠 Chat Memory
    - 📄 PDF RAG
    - 🔎 Semantic Search
    - ⚡ Streaming Responses
    - 🌡️ Temperature Control
    - 🐍 Python + Streamlit
    - 🦙 Ollama Local LLM
    """)


# ============================================================
# CHECK OLLAMA
# ============================================================

if not check_ollama_running():

    st.error("❌ Ollama is not running.")

    st.info(
        "Open Terminal and run:\n\n"
        "ollama serve"
    )

    st.stop()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_prompt = st.chat_input(
    "Ask about Python, AI, ML, LLM, RAG or your PDF... 🤖"
)


# ============================================================
# USER QUESTION
# ============================================================

if user_prompt:

    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(user_prompt)


    # ========================================================
    # DETERMINE MESSAGE TYPE
    # ========================================================

    greeting_message = is_greeting(user_prompt)

    casual_message = is_casual_message(user_prompt)


    # ========================================================
    # RETRIEVE PDF CONTEXT
    # ========================================================

    document_context = ""

    # Do NOT search PDF for greetings or casual messages
    if not greeting_message and not casual_message:

        if st.session_state.documents:

            try:

                relevant_chunks = retrieve_documents(
                    user_prompt,
                    top_k=3,
                    minimum_score=0.30
                )

                if relevant_chunks:

                    document_context = (
                        "\n\n".join(
                            relevant_chunks
                        )
                    )

            except Exception as e:

                st.warning(
                    f"Could not search PDF: {e}"
                )


    # ========================================================
    # BUILD FINAL SYSTEM PROMPT
    # ========================================================

    final_system_prompt = system_prompt


    # --------------------------------------------------------
    # PDF CONTEXT EXISTS
    # --------------------------------------------------------

    if document_context:

        final_system_prompt += f"""

IMPORTANT DOCUMENT CONTEXT:

A PDF document has been uploaded and indexed.

The user may be asking about this document.

Use the retrieved document context below when
answering questions related to the PDF.

DOCUMENT CONTEXT:
-------------------------
{document_context}
-------------------------

Rules:

1. Use the document context when the question is
   related to the uploaded PDF.

2. Do not claim that no PDF was uploaded.

3. If the answer is not present in the retrieved
   document context, say that the information was
   not found in the uploaded PDF.

4. Do not invent information from the document.

5. If the user asks a normal question unrelated
   to the PDF, answer normally.
"""


    # --------------------------------------------------------
    # NO PDF CONTEXT
    # --------------------------------------------------------

    else:

        final_system_prompt += """

There is currently no relevant PDF context for this question.

Answer the user's question normally using your general knowledge.

If the user is greeting you, respond naturally and briefly.

Do not mention the PDF unless the user specifically asks about it.
"""


    # ========================================================
    # BUILD OLLAMA MESSAGES
    # ========================================================

    messages_for_ollama = [
        {
            "role": "system",
            "content": final_system_prompt
        }
    ]

    messages_for_ollama.extend(
        st.session_state.messages
    )


    # ========================================================
    # AI RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""


        try:

            payload = {
                "model": model,

                "messages": messages_for_ollama,

                "stream": True,

                "options": {
                    "temperature": temperature
                }
            }


            # =================================================
            # SEND REQUEST TO OLLAMA
            # =================================================

            with requests.post(
                OLLAMA_CHAT_URL,
                json=payload,
                stream=True,
                timeout=180
            ) as response:

                response.raise_for_status()


                # =============================================
                # STREAM RESPONSE
                # =============================================

                for line in response.iter_lines():

                    if line:

                        data = line.decode("utf-8")

                        chunk = json.loads(data)


                        # =====================================
                        # GET RESPONSE CONTENT
                        # =====================================

                        if (
                            "message" in chunk
                            and "content"
                            in chunk["message"]
                        ):

                            content = (
                                chunk["message"]["content"]
                            )

                            full_response += content


                            response_placeholder.markdown(
                                full_response + "▌"
                            )


                        # =====================================
                        # RESPONSE COMPLETE
                        # =====================================

                        if chunk.get("done", False):

                            break


            # =================================================
            # FINAL RESPONSE
            # =================================================

            response_placeholder.markdown(
                full_response
            )


        # ====================================================
        # HTTP ERROR
        # ====================================================

        except requests.exceptions.HTTPError as e:

            full_response = (
                f"❌ HTTP Error: {str(e)}\n\n"
                f"Make sure the model is installed:\n\n"
                f"ollama pull {model}"
            )

            response_placeholder.error(
                full_response
            )


        # ====================================================
        # CONNECTION ERROR
        # ====================================================

        except requests.exceptions.ConnectionError:

            full_response = (
                "❌ Could not connect to Ollama.\n\n"
                "Run:\n\n"
                "ollama serve"
            )

            response_placeholder.error(
                full_response
            )


        # ====================================================
        # GENERAL ERROR
        # ====================================================

        except Exception as e:

            full_response = (
                f"❌ Error: {str(e)}"
            )

            response_placeholder.error(
                full_response
            )


    # ========================================================
    # SAVE RESPONSE TO MEMORY
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )