from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ==========================================
# 1. LOAD VECTOR DATABASE
# ==========================================

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


# ==========================================
# 2. LOAD LOCAL LLM
# ==========================================

print("\nLoading local AI model...")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto"
)

print("Local AI model loaded.")


# ==========================================
# 3. FUNCTION TO GENERATE ANSWER
# ==========================================

def generate_answer(question):

    # Search DRDO knowledge base
    results = vectorstore.similarity_search(
        question,
        k=5
    )

    # Combine retrieved documents
    context = ""

    for i, doc in enumerate(results):

        context += f"\n--- Document {i + 1} ---\n"
        context += doc.page_content[:2500]

    # Prompt
    prompt = f"""
You are a DRDO Defence Information Assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I could not find this information in the available DRDO documents."

Do not invent facts.

Give a clear and concise answer.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful DRDO information assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=True
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer, results


# ==========================================
# 4. CHAT LOOP
# ==========================================

print("\n======================================")
print("      DRDO DEFENCE CHATBOT")
print("======================================")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    answer, results = generate_answer(question)

    print("\nDRDO Assistant:")
    print(answer)

    print("\nSources:")

    for i, doc in enumerate(results):

        print(
            f"{i + 1}. "
            f"{doc.metadata.get('source', 'Unknown')}"
        )

    print("\n" + "=" * 50 + "\n")