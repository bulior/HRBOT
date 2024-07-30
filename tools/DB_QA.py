import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory for the QA database
current_dir = os.path.dirname(os.path.abspath(__file__))
qa_persistent_directory = os.path.join(current_dir, "../db", "chroma_db_QA")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

qa_db = Chroma(persist_directory=qa_persistent_directory, embedding_function=embeddings)

# Function to retrieve all documents from the QA database
def get_all_documents(qa_db):
    all_docs = []
    try:
        collection = qa_db._collection
        # Get all document IDs
        all_ids = collection.get_all_ids()
        print(f"All IDs: {all_ids}")
        # Retrieve all documents by their IDs
        for doc_id in all_ids:
            doc = collection.get(ids=[str(doc_id)])
            if doc:
                all_docs.extend(doc)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
    return all_docs

# Retrieve all documents
qa_docs = get_all_documents(qa_db)

# Check the structure of the retrieved documents
print(f"Retrieved documents: {qa_docs}")

# Create a DataFrame to display the data
data = {
    "Date": [],
    "Question": [],
    "Answer": [],
    "Correct": []
}

for doc in qa_docs:
    content = doc.get('page_content', '').split("\n")
    question = content[0] if content else "No Content"
    answer = "\n".join(content[1:]) if len(content) > 1 else "No Answer"
    date = doc.get("metadata", {}).get("date", "No Date")
    correct_answer = doc.get("metadata", {}).get("correctAnswer", "Unknown")
    
    data["Date"].append(date)
    data["Question"].append(question)
    data["Answer"].append(answer)
    data["Correct"].append(correct_answer)

df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# If you want to save the DataFrame to a CSV file
csv_path = os.path.join(current_dir, "qa_overview.csv")
df.to_csv(csv_path, index=False)
print(f"Übersicht wurde als CSV gespeichert: {csv_path}")