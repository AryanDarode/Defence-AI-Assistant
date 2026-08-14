import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from google import genai


# ==============================
# LOAD ENVIRONMENT
# ==============================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    exit()

client = genai.Client(api_key=api_key)

print("Gemini connected.")


# ==============================
# LOAD VECTOR DATABASE
# ==============================

VECTORSTORE_PATH = "vectorstore"

print("Loading vector database...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)

print("Vector database loaded.")


# ==============================
# ASK QUESTION
# ==============================

question = input("\nAsk a question about Defence: ")


# ==============================
# RETRIEVAL
# ==============================

results_with_scores = vectorstore.similarity_search_with_score(
    question,
    k=8
)

MAX_DISTANCE = 0.95

results = []

for doc, score in results_with_scores:

    if score <= MAX_DISTANCE:
        results.append(doc)

if not results:
    print("\nI could not find relevant information in the provided Defence dataset.")
    exit()
# ==============================
# BUILD CONTEXT
# ==============================

context = ""

for i, doc in enumerate(results):

    source = doc.metadata.get("source", "Unknown source")

    context += f"""
SOURCE {i + 1}:
{source}

CONTENT:
{doc.page_content}

-------------------------
"""


# ==============================
# GEMINI PROMPT
# ==============================

prompt = f"""
You are a Defence-domain AI assistant.

Your task is to answer the user's question using ONLY
the retrieved information from the provided Defence dataset.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent or assume facts.
3. If the retrieved documents do not contain enough
   information to answer the question, say:

   "I could not find sufficient information in the
   provided Defence dataset."

4. Give a concise and factual answer.
5. Base every important claim on the retrieved documents.

USER QUESTION:
{question}

RETRIEVED DEFENCE DOCUMENTS:

{context}
"""


# ==============================
# GENERATE ANSWER
# ==============================

print("\nGenerating answer...")

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)


# ==============================
# DISPLAY ANSWER
# ==============================

print("\n")
print("=" * 45)
print("          DEFENCE AI ASSISTANT")
print("=" * 45)

print("\nANSWER:")
print(response.text)


# ==============================
# SOURCES
# ==============================

print("\n")
print("=" * 45)
print("              SOURCES")
print("=" * 45)

seen = set()

for doc in results:

    source = doc.metadata.get("source", "Unknown")

    if source not in seen:
        print("•", source)
        seen.add(source)