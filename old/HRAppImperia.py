import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import os
import faiss
import numpy as np

# Funktion zum Einlesen der Dokumente aus dem Ordner "docs" mit Metadaten
def load_documents_with_metadata(directory="docs"):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") or filename.endswith(".pdf"):
            content = ""
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                        content = file.read()
                except Exception as e:
                    st.write(f"Fehler beim Einlesen der Datei {filename}: {e}")
            elif filename.endswith(".pdf"):
                try:
                    with fitz.open(os.path.join(directory, filename)) as doc:
                        for page in doc:
                            content += page.get_text()
                except Exception as e:
                    st.write(f"Fehler beim Einlesen der Datei {filename}: {e}")
            if content:
                documents.append({"filename": filename, "content": content})
    return documents

# Funktion, um Dokumente in kleinere Abschnitte aufzuteilen
def split_into_chunks(text, chunk_size=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Laden der Dokumente mit Metadaten
documents = load_documents_with_metadata()
all_chunks = []
metadata = []

# Aufteilen der Dokumente in kleinere Abschnitte und Metadaten speichern
for doc in documents:
    chunks = split_into_chunks(doc["content"])
    all_chunks.extend(chunks)
    metadata.extend([doc["filename"]] * len(chunks))

# Erstellen der Embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name)
corpus_embeddings = embedder.encode(all_chunks, convert_to_tensor=True)

# Index für Faiss erstellen
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings.cpu().numpy())

# Laden des T5-Modells und Tokenizers
t5_model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Streamlit App
st.title("Frage-Antwort-System mit Quellenangaben")

# Benutzerfrageingabe
question = st.text_input("Stelle eine Frage:")

# Wenn eine Frage eingegeben wurde, generiere eine Antwort
if question:
    with st.spinner('Generiere Antwort...'):
        # Erstellen des Frage-Embeddings
        question_embedding = embedder.encode(question, convert_to_tensor=True)

        # Überprüfen der Form des Frage-Embeddings
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.unsqueeze(0)

        # Sicherstellen, dass das Frage-Embedding die korrekte Form hat
        if len(question_embedding.shape) == 2 and question_embedding.shape[0] == 1:
            question_embedding = question_embedding[0].cpu().numpy()

            # Suche der ähnlichsten Abschnitte mit Faiss
            top_k = 5
            D, I = index.search(np.array([question_embedding]), top_k)

            # Zusammensetzen des Kontexts aus den ähnlichsten Abschnitten
            context = " ".join([all_chunks[i] for i in I[0]])
            sources = [metadata[i] for i in I[0]]

            # Generieren der Antwort
            input_text = f"question: {question} context: {context}"
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Anzeige der Antwort und Quellen
            st.write("Antwort:", answer)
            st.write("Quelle(n):")
            for source in set(sources):
                st.write(f"Dokument: {source}")
        else:
            st.write("Fehler beim Erstellen des Frage-Embeddings. Bitte versuche es erneut.")