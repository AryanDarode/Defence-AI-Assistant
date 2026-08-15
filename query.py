import os
import re

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_PATH = "vectorstore"

TOP_K = 15


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    print("ERROR: GEMINI_API_KEY not found")

    exit()


client = genai.Client(
    api_key=api_key
)

print("Gemini connected.")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD CHROMA
# ============================================================

print("Loading Chroma vector database...")

vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)

print("Vector database loaded.")


# ============================================================
# KEYWORD SCORING
# ============================================================

def keyword_score(question, text):

    question_lower = question.lower()

    text_lower = text.lower()

    score = 0


    # Important phrases

    important_phrases = [

        "dual thrust",

        "edb",

        "propellant",

        "solid propellant",

        "rocket motor",

        "double base",

        "propulsion"

    ]


    for phrase in important_phrases:

        if phrase in question_lower:

            if phrase in text_lower:

                score += 10


    # Individual words

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        question_lower
    )


    for word in words:

        if len(word) > 2:

            if word in text_lower:

                score += 1


    return score


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(question):

    # Semantic retrieval

    semantic_results = vectorstore.similarity_search(
        question,
        k=TOP_K
    )


    scored_results = []


    for doc in semantic_results:

        score = keyword_score(
            question,
            doc.page_content
        )


        scored_results.append(
            (score, doc)
        )


    # Highest keyword relevance first

    scored_results.sort(
        key=lambda x: x[0],
        reverse=True
    )


    return [
        doc
        for score, doc in scored_results[:8]
    ]


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    print("\nSearching DRDO knowledge base...")

    results = retrieve_documents(question)


    if not results:

        return (
            "I could not find sufficient information "
            "in the provided DRDO documents.",
            []
        )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = ""


    for i, doc in enumerate(results):

        context += f"""

================ DOCUMENT {i + 1} ================

SOURCE:
{doc.metadata.get("source", "DRDO document")}

CONTENT:

{doc.page_content}

====================================================
"""


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""

You are a DRDO Defence Information Assistant.

Your job is to answer the user's question using ONLY
the information contained in the retrieved DRDO documents.

IMPORTANT RULES:

1. Do NOT use outside knowledge.

2. Do NOT invent facts.

3. Do NOT assume that an abbreviation means something
   unless the retrieved documents explicitly establish it.

4. If the documents contain related information but
   do not completely answer the question, clearly explain
   what IS available and what is NOT available.

5. Never hallucinate.

6. Give a concise and easy-to-understand answer.

7. Mention the relevant DRDO system/product when possible.

8. If appropriate, mention the page number shown in
   the retrieved content.

USER QUESTION:

{question}


RETRIEVED DRDO DOCUMENTS:

{context}


ANSWER:
"""


    # ========================================================
    # GEMINI
    # ========================================================

    response = client.models.generate_content(

        model="gemini-3.6-flash",

        contents=prompt
    )


    return response.text, results


# ============================================================
# CHATBOT
# ============================================================

print("\n")
print("=" * 60)
print("              DRDO DEFENCE AI")
print("=" * 60)

print(
    "Ask questions about the available DRDO documents."
)

print(
    "Type 'exit' to stop."
)

print("=" * 60)


while True:

    question = input("\nYou: ").strip()


    if not question:

        continue


    if question.lower() == "exit":

        print("\nChatbot stopped.")

        break


    answer, results = generate_answer(
        question
    )


    # ========================================================
    # ANSWER
    # ========================================================

    print("\nDRDO Assistant:")
    print(answer)


    # ========================================================
    # SOURCES
    # ========================================================

    print("\nSources:")


    seen = set()


    for doc in results:

        source = doc.metadata.get(
            "source",
            "DRDO document"
        )


        if source not in seen:

            print("•", source)

            seen.add(source)


    print("\n" + "=" * 60)