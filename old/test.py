import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Funktion zur Token-Zählung
def count_tokens(text):
    # Einfache Annäherung: Ein Token ist ungefähr 4 Zeichen (dies ist eine Annäherung für GPT-3/4)
    return len(text) // 4

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  #text-embedding-3-large text-embedding-3-small

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo-0125")  #gpt-3.5-turbo-0125 gpt-4o

# Streamlit app
st.image('img/RH.png', width=200) #use_column_width=True,
st.title("Wie kann ich Ihnen helfen?")

query = st.text_input("Stellen Sie eine Frage zur Weiterbildung von Führungskräften:")

if st.button("Frage stellen"):
    if query:
        with st.spinner("Bitte warten Sie, die Anfrage wird bearbeitet..."):
            # Retrieve relevant documents based on the query
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Anzahl der zu suchenden Dokumente anpassen
            relevant_docs = retriever.invoke(query)

            # Display the relevant results with metadata
            st.markdown('<p style="color:lightblue;">\nFolgendes konnte ich in meinen Dokumenten finden:</p>', unsafe_allow_html=True)
            for i, doc in enumerate(relevant_docs, 1):
                st.markdown(f'<p style="color:gray;">Document {i} Title:\n{doc.metadata.get("title", "No Title")}</p>', unsafe_allow_html=True)

            # Funktion zur Erstellung kombinierter Eingaben
            def create_combined_input(query, docs):
                combined_input = (
                    "Here are some documents that might help answer the question: "
                    + query
                    + "\n\nRelevant Documents:\n"
                )
                tokens = count_tokens(combined_input)
                max_tokens = 4096 - tokens  # Maximum Tokens - Tokens der bisherigen Anfrage
                doc_chunks = []

                for doc in docs:
                    doc_text = doc.page_content
                    doc_tokens = count_tokens(doc_text)

                    # Split the document into smaller chunks if it exceeds max_tokens
                    if doc_tokens > max_tokens:
                        doc_chunks.append(doc_text[:max_tokens * 4])
                        continue

                    if tokens + doc_tokens <= max_tokens:
                        combined_input += doc_text + "\n\n"
                        tokens += doc_tokens
                    else:
                        doc_chunks.append(combined_input)
                        combined_input = doc_text + "\n\n"
                        tokens = doc_tokens

                if combined_input:
                    doc_chunks.append(combined_input)

                return doc_chunks

            # Define the messages for the model
            responses = []
            doc_chunks = create_combined_input(query, relevant_docs)

            for chunk in doc_chunks:
                messages = [
                    SystemMessage(content="You are a helpful assistant. Antworte immer in der Sprache wie die Frage gestellt wurde."),
                    HumanMessage(content=chunk),
                ]

                # Invoke the model with the combined input
                result = model.invoke(messages)
                responses.append(result.content)
                
                # Wartezeit zwischen den Anfragen, um die Token-Limitierung zu umgehen
                time.sleep(5)  # Wartezeit in Sekunden

            # Display the full result and content only
            st.markdown('<p style="color:lightblue;">\nZusammengefasste Antworten:</p>', unsafe_allow_html=True)
            for response in responses:
                st.write(response)
    else:
        st.write("Bitte geben Sie eine Frage ein.")