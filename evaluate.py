from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


print("Loading vector database...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

print("Vector database loaded.")


questions = [
    "What is DRDO?",
    "What are the major technology areas of DRDO?",
    "What is the role of DRDO in defence research?",
    "What are DRDO laboratories?",
    "What defence technologies are developed by DRDO?"
]


for question in questions:

    print("\n")
    print("=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    results = vectorstore.similarity_search_with_score(
        question,
        k=5
    )

    for i, (doc, score) in enumerate(results):

        print(f"\nRESULT {i + 1}")
        print("-" * 60)

        print("SCORE:", round(score, 4))

        print(
            "SOURCE:",
            doc.metadata.get("source", "Unknown")
        )

        text = doc.page_content[:400]

        print(
            "CONTENT:",
            text.replace("\n", " ")
        )