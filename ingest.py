import os
import shutil
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pypdf import PdfReader


# ============================================================
# PATHS
# ============================================================

CSV_PATH = "data/raw/DRDO.csv"
PDF_DIR = "data/raw/india/DRDO"
VECTORSTORE_PATH = "vectorstore"


print("============================================")
print("       DRDO PDF → CHROMA INGESTION")
print("============================================")


# ============================================================
# 1. CHECK FILES
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}"
    )

if not os.path.exists(PDF_DIR):
    raise FileNotFoundError(
        f"PDF directory not found: {PDF_DIR}"
    )


# ============================================================
# 2. LOAD CSV
# ============================================================

print("\n[1/6] Loading DRDO CSV...")

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8",
    low_memory=False,
    on_bad_lines="skip"
)

print(f"CSV rows: {len(df)}")


# ============================================================
# 3. READ PDF FILES
# ============================================================

print("\n[2/6] Extracting text from PDFs...")

documents = []

successful = 0
failed = 0
total_pages = 0


for index, row in df.iterrows():

    local_path = str(row.get("local_path", "")).strip()

    if not local_path or local_path == "nan":

        print(f"Skipping row {index}: no local_path")
        failed += 1
        continue


    # Convert CSV path to current OS path
    pdf_path = local_path.replace("\\", os.sep)


    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if not os.path.exists(pdf_path):

        filename = os.path.basename(
            local_path.replace("\\", "/")
        )

        pdf_path = os.path.join(
            PDF_DIR,
            filename
        )


    if not os.path.exists(pdf_path):

        print(
            f"WARNING: PDF not found: {pdf_path}"
        )

        failed += 1
        continue


    # --------------------------------------------------------
    # Extract PDF text
    # --------------------------------------------------------

    try:

        reader = PdfReader(pdf_path)

        pdf_text = ""

        for page_number, page in enumerate(reader.pages):

            try:

                page_text = page.extract_text()

                if page_text:
                    pdf_text += (
                        f"\n\n"
                        f"[Page {page_number + 1}]\n"
                        f"{page_text}"
                    )

            except Exception as e:

                print(
                    f"Warning: Could not read page "
                    f"{page_number + 1} in {pdf_path}: {e}"
                )


        pdf_text = pdf_text.strip()


        if not pdf_text:

            print(
                f"WARNING: No text extracted: "
                f"{os.path.basename(pdf_path)}"
            )

            failed += 1
            continue


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "source": "DRDO",
            "document_id": str(
                row.get("document_id", "")
            ),
            "title": str(
                row.get("title", "")
            ),
            "category": str(
                row.get("category", "")
            ),
            "url": str(
                row.get("url", "")
            ),
            "pdf_path": pdf_path
        }


        document = Document(
            page_content=pdf_text,
            metadata=metadata
        )


        documents.append(document)

        successful += 1
        total_pages += len(reader.pages)


        print(
            f"[{index + 1}/{len(df)}] "
            f"OK: {os.path.basename(pdf_path)} "
            f"({len(reader.pages)} pages)"
        )


    except Exception as e:

        print(
            f"ERROR reading {pdf_path}: {e}"
        )

        failed += 1


# ============================================================
# EXTRACTION SUMMARY
# ============================================================

print("\n--------------------------------------------")
print("PDF EXTRACTION SUMMARY")
print("--------------------------------------------")

print(f"PDFs successfully read : {successful}")
print(f"PDFs failed            : {failed}")
print(f"Total pages            : {total_pages}")


if not documents:

    raise RuntimeError(
        "\nNo PDF documents were successfully extracted.\n"
        "Cannot create vector database."
    )


# ============================================================
# 4. CHUNK DOCUMENTS
# ============================================================

print("\n[3/6] Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(
    documents
)

print(f"Total chunks created: {len(chunks)}")


if not chunks:

    raise RuntimeError(
        "No chunks were created."
    )


# ============================================================
# 5. LOAD EMBEDDINGS
# ============================================================

print("\n[4/6] Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 6. DELETE OLD VECTORSTORE
# ============================================================

print("\n[5/6] Rebuilding vector database...")

if os.path.exists(VECTORSTORE_PATH):

    shutil.rmtree(VECTORSTORE_PATH)

    print("Old vectorstore deleted.")


# ============================================================
# CREATE CHROMA
# ============================================================

print("\n[6/6] Creating Chroma vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTORSTORE_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\n============================================")
print("       INGESTION COMPLETED SUCCESSFULLY")
print("============================================")

print(f"Documents : {len(documents)}")
print(f"Chunks    : {len(chunks)}")
print(f"Pages     : {total_pages}")
print(f"Vector DB : {VECTORSTORE_PATH}")

print("============================================")