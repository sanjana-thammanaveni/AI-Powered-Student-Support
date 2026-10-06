from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


COLLEGE_FOLDER = Path("college")


def load_documents():
    documents = []

    root_file = Path("college_info.txt")

    if root_file.exists():
        text = root_file.read_text(encoding="utf-8")

        if text.strip():
            documents.append({
                "source": root_file.name,
                "content": text
            })

    if COLLEGE_FOLDER.exists():

        for file in COLLEGE_FOLDER.glob("*.txt"):

            text = file.read_text(encoding="utf-8")

            if text.strip():
                documents.append({
                    "source": file.name,
                    "content": text
                })

    return documents


def split_text(text):

    # Split based on sections and paragraphs
    sections = text.split("\n\n")

    chunks = []

    for section in sections:

        section = section.strip()

        if section:
            chunks.append(section)

    return chunks


def create_chunks():

    documents = load_documents()

    chunks = []

    for document in documents:

        document_chunks = split_text(
            document["content"]
        )

        for chunk in document_chunks:

            chunks.append({
                "source": document["source"],
                "content": chunk
            })

    return chunks


def retrieve_documents(question, top_k=3):

    chunks = create_chunks()

    if not chunks:
        return []

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(
        texts + [question]
    )

    similarities = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        if similarities[index] > 0:

            results.append({

                "source": chunks[index]["source"],

                "content": chunks[index]["content"],

                "score": similarities[index]

            })

    return results