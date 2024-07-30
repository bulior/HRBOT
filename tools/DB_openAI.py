import os
import json
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "../docs/txt")
db_dir = os.path.join(current_dir, "../db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")
metadata_file = os.path.join(db_dir, "metadata.json")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}") #chromaDB

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Check if the Chroma vector store already exists
if os.path.exists(persistent_directory):
    print("Vector store already exists. Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
else:
    print("Persistent directory does not exist. Initializing vector store...")
    db = None

# Ensure the books directory exists
if not os.path.exists(books_dir):
    raise FileNotFoundError(
        f"The directory {books_dir} does not exist. Please check the path."
    )

# Load existing document metadata
existing_sources = set()
if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        existing_sources = set(json.load(f))

# List all text files in the directory
book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

# Display information about found files and their sizes
print("\n--- Found Files Information ---")
for book_file in book_files:
    file_path = os.path.join(books_dir, book_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"Found file: {book_file} with {len(content)} characters")

# Read the text content from each file and store it with metadata
documents = []
new_files = {}  # Dictionary to keep track of newly added files and their chunk sizes
upload_date = datetime.now().strftime("%Y-%m-%d")  # Get the current date
for book_file in book_files:
    if book_file in existing_sources:
        print(f"Skipping already known file: {book_file}")
        continue
    file_path = os.path.join(books_dir, book_file)
    loader = TextLoader(file_path)
    book_docs = loader.load()
    for doc in book_docs:
        # Add metadata to each document indicating its source and upload date
        doc.metadata = {
            "source": book_file,
            "title": os.path.splitext(book_file)[0],
            "upload_date": upload_date
        }
        documents.append(doc)
    existing_sources.add(book_file)

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
all_chunks = text_splitter.split_documents(documents)

# Calculate chunks per document
chunk_counts = {book_file: 0 for book_file in book_files}
for chunk in all_chunks:
    source_file = chunk.metadata["source"]
    if source_file in chunk_counts:
        chunk_counts[source_file] += 1

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(all_chunks)}")


if all_chunks:
    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create or update the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    if db:
        db.add_documents(all_chunks)
    else:
        db = Chroma.from_documents(
            all_chunks, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")
    #db.persist() #not needed anymore

    # Save the updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(list(existing_sources), f)

    # Print newly added documents and their chunk sizes
    print("\n--- Newly Added Documents ---")
    for file, size in chunk_counts.items():
        if size > 0:  # Only print files that were newly added
            print(f"Added: {file} with {size} chunks")
else:
    print("No new documents to add.")