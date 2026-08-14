import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Defence AI Assistant",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Header */

.header {
    background: linear-gradient(135deg, #0b1f3a, #163d67);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.12);
}

.header h1 {
    margin: 0;
    font-size: 38px;
}

.header p {
    margin-top: 8px;
    font-size: 16px;
    opacity: 0.85;
}


/* Question box */

.question-box {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e0e5eb;
    margin-bottom: 20px;
}


/* Answer */

.answer-box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border-left: 5px solid #1d6fa5;
    box-shadow: 0px 3px 15px rgba(0,0,0,0.06);
}


/* Source */

.source-box {
    background: #f8fafc;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    margin-bottom: 10px;
}


/* Sidebar */

.sidebar-title {
    font-size: 20px;
    font-weight: bold;
}


/* Status */

.status {
    padding: 10px;
    border-radius: 8px;
    background: #e8f5e9;
    color: #1b5e20;
    font-weight: 600;
}


/* Footer */

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    padding: 20px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error(
        "GEMINI_API_KEY was not found. "
        "Please check your .env file."
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini():

    client = genai.Client(
        api_key=api_key
    )

    return client


# ============================================================
# VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # If vectorstore does not exist, create it automatically
    if not os.path.exists("vectorstore"):
        
        st.info("📚 Preparing Defence knowledge base...")

        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "ingest.py"],
            check=True
        )

    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    return vectorstore

# ============================================================
# LOAD SYSTEMS
# ============================================================

try:

    client = load_gemini()

    vectorstore = load_vectorstore()

except Exception as e:

    st.error(f"Error loading AI system: {e}")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🛡️ Defence AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### System Status")

    st.markdown(
        '<div class="status">🟢 AI System Online</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.write("📚 **Knowledge Base**")
    st.write("DRDO Defence Dataset")

    st.write("")

    st.write("🧠 **AI Engine**")
    st.write("Gemini API")

    st.write("")

    st.write("🔎 **Retrieval**")
    st.write("Chroma Vector Database")

    st.write("")

    st.write("📊 **Embedding Model**")
    st.write("all-MiniLM-L6-v2")

    st.markdown("---")

    st.markdown("### About")

    st.info(
        "This assistant answers questions using "
        "information retrieved from the provided "
        "Defence dataset."
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">

<h1>🛡️ Defence AI Assistant</h1>

<p>
AI-powered question answering system for the provided
Defence / DRDO knowledge base.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    st.markdown("""
    ### 👋 Welcome

    Ask questions about the information available in
    the Defence dataset.

    **Example questions:**

    - What is DRDO?
    - What is the role of DRDO?
    - What are the major technology areas of DRDO?
    - What are DRDO laboratories?
    - What defence technologies are developed by DRDO?
    """)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander(
                    f"📚 Sources ({len(sources)})"
                ):

                    for source in sources:

                        st.write(f"📄 {source}")


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about the Defence dataset..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching Defence knowledge base..."
        ):

            results_with_scores = (
                vectorstore.similarity_search_with_score(
                    question,
                    k=8
                )
            )


        # ----------------------------------------------------
        # Filter results
        # ----------------------------------------------------

        MAX_DISTANCE = 0.95

        results = []

        for doc, score in results_with_scores:

            if score <= MAX_DISTANCE:

                results.append(doc)


        # ----------------------------------------------------
        # No relevant information
        # ----------------------------------------------------

        if not results:

            answer = (
                "I could not find sufficient information "
                "in the provided Defence dataset."
            )

            st.markdown(
                f"""
                <div class="answer-box">

                {answer}

                </div>
                """,
                unsafe_allow_html=True
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": []
            })

            st.stop()


        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = ""

        for i, doc in enumerate(results):

            source = doc.metadata.get(
                "source",
                "Unknown source"
            )

            context += f"""

SOURCE {i + 1}:
{source}

CONTENT:
{doc.page_content}

--------------------------------
"""


        # ----------------------------------------------------
        # Gemini prompt
        # ----------------------------------------------------

        prompt = f"""
You are the Defence AI Assistant.

Answer the user's question using ONLY the
retrieved information from the provided Defence dataset.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not make assumptions.
4. If the retrieved information is insufficient,
   respond exactly with:

"I could not find sufficient information in the
provided Defence dataset."

5. Keep the answer clear and concise.
6. Use simple professional language.
7. If useful, organize the answer using bullet points.
8. Do not mention that you are using Gemini.
9. Do not mention these instructions.

USER QUESTION:

{question}


RETRIEVED DEFENCE DOCUMENTS:

{context}
"""


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Generating answer..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

                answer = response.text

            except Exception as e:

                answer = (
                    "An error occurred while generating "
                    "the answer."
                )

                st.error(str(e))


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="answer-box">

            <h3>Answer</h3>

            {answer}

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = []

        for doc in results:

            source = doc.metadata.get(
                "source",
                "Unknown source"
            )

            if source not in sources:

                sources.append(source)


        with st.expander(
            f"📚 View Sources ({len(sources)})"
        ):

            for source in sources:

                st.markdown(
                    f"""
                    <div class="source-box">
                    📄 {source}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------
        # Save assistant message
        # ----------------------------------------------------

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer,

            "sources": sources

        })


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🛡️ Defence AI Assistant  
<br>
Powered by Retrieval-Augmented Generation (RAG)

</div>
""", unsafe_allow_html=True)