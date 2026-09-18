import os
import sys
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2

# Force console to use UTF-8 to support emojis
sys.stdout.reconfigure(encoding='utf-8')

# ------------------------- CONFIGURATION -------------------------
KB_DIR = "./daraz_knowledge_base"
FAISS_DIR = "./faiss_index"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts raw text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 15:  # Ignore trivial fragments
            chunks.append(chunk)
    return chunks


def main():
    print("🚀 Starting Knowledge Base Ingestion...")
    os.makedirs(FAISS_DIR, exist_ok=True)

    print(f"📦 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    metadata = []
    embeddings = []

    # Process all PDFs organized by department
    for department in os.listdir(KB_DIR):
        dept_path = os.path.join(KB_DIR, department)
        if not os.path.isdir(dept_path):
            continue

        print(f"📂 Processing department: {department}")
        for filename in os.listdir(dept_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(dept_path, filename)
                print(f"  📄 Reading: {filename}")

                text = extract_text_from_pdf(pdf_path)
                chunks = chunk_text(text)
                print(f"     ✂️ Created {len(chunks)} chunks")

                for chunk in chunks:
                    metadata.append({
                        "text": chunk,
                        "department": department,
                        "source_file": filename
                    })
                    embeddings.append(model.encode(chunk))

    if not embeddings:
        print("❌ No text chunks found. Check your PDF files and directory structure.")
        return

    # Convert to numpy array and normalize for Cosine Similarity (Inner Product)
    embeddings_array = np.array(embeddings).astype("float32")
    dimension = embeddings_array.shape[1]

    print(f"🔍 Building FAISS index (dimension: {dimension})...")
    index = faiss.IndexFlatIP(dimension)
    # Crucial for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)

    # Save artifacts
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Ingestion complete!")
    print(f"   - Index saved to: {INDEX_PATH}")
    print(f"   - Metadata saved to: {METADATA_PATH}")
    print(f"   - Total chunks indexed: {len(metadata)}")


if __name__ == "__main__":
    main()
