import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata") #chroma_db_openai, chroma_db_with_metadata

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  #text-embedding-3-large text-embedding-3-small

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo-0125")  #gpt-4o

# Streamlit app
st.image('img/RH.png', width=200) #use_column_width=True,
st.title("Wie kann ich Ihnen helfen?")

query = st.text_input("Stellen Sie eine Frage zur Weiterbildung von Führungskräften:")

if st.button("Frage stellen"):
    if query:
        with st.spinner("Bitte warten Sie, die Anfrage wird bearbeitet..."):
            # Retrieve relevant documents based on the query
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
            relevant_docs = retriever.invoke(query)

            # Display the relevant results with metadata
            #st.write("\n--- Informationen in folgenden Dokumenten gefunden ---")
            st.markdown('<p style="color:lightblue;">\nFolgendes konnte ich in meinen Dokumenten finden:</p>', unsafe_allow_html=True)
            for i, doc in enumerate(relevant_docs, 5):
                st.markdown(f'<p style="color:gray;">Document {i} Title:\n{doc.metadata.get("title", "No Title")}</p>', unsafe_allow_html=True)

            # Combine the query and the relevant document contents
            combined_input = (
                "Here are some documents that might help answer the question: "
                + query
                + "\n\nRelevant Documents:\n"
                + "\n\n".join([doc.page_content for doc in relevant_docs])
                + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with: Ich bin mir nicht sicher, bitte wenden Sie sich an die zuständige Abteilung'."
            )

            # Define the messages for the model
            messages = [
                SystemMessage(content="You are a helpful assistant. Antworte immer in der Sprache wie die Frage gestellt wurde."),
                HumanMessage(content=combined_input),
            ]

            # Invoke the model with the combined input
            result = model.invoke(messages)

            # Display the full result and content only
            #st.write("\nFolgendes konnte ich in meinen Dokumenten finden:")
            st.write(result.content)
    else:
        st.write("Bitte geben Sie eine Frage ein.")