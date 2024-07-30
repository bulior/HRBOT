##
## 2024-07-25 v3 - Vorgestellt in Runde 2 mit Karl, Ella Seel und XX
##
import os
import streamlit as st
import subprocess
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#import getpass

# PDF to TXT
from tools.PDFtoTXT import convert_pdfs_in_folder

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")
docs_directory = os.path.join(current_dir, "docs")

# Ensure the docs directory exists
os.makedirs(docs_directory, exist_ok=True)

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o")

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Streamlit app
st.set_page_config(page_title='RH Bot',  layout = 'wide', initial_sidebar_state = 'auto')

# Define username and password
USERNAME = "RH"
PASSWORD = "5555"

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login function
def login():
    if st.session_state.username == USERNAME and st.session_state.password == PASSWORD:
        st.session_state.authenticated = True
    else:
        st.error("Invalid username or password")

    # Show login screen if not authenticated
    if not st.session_state.authenticated:
        st.title("Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=login)
    else:
    st.title("💬 HR Chatbot")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "qa_data" not in st.session_state:
        st.session_state.qa_data = []
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = 1
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False

    # Get the PC username
    pc_username = "APP" #getpass.getuser()

    # Function to handle user input
    def handle_user_input():
        user_input = st.session_state.user_input
        result = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(SystemMessage(content=result["answer"]))
        
        # Log the QA data
        qa_entry = {
            "chat_id": st.session_state.chat_id,
            "user": pc_username,
            "date": datetime.now().isoformat(),
            "question": user_input,
            "answer": result["answer"],
            "correctAnswer": None  # Will be updated based on user feedback
        }
        st.session_state.qa_data.append(qa_entry)
        
        # Save Q&A data immediately to QA.json
        save_all_qa_data()

        # Clear the input box after processing
        st.session_state.user_input = ""

    # Function to save QA data to JSON file
    def save_qa_data():
        qa_file_path = os.path.join(current_dir, "db/QA_feedback.json")
        
        # Load existing data
        if os.path.exists(qa_file_path):
            with open(qa_file_path, "r") as qa_file:
                existing_data = json.load(qa_file)
        else:
            existing_data = []

        # Append new data
        existing_data.extend(st.session_state.qa_data)
        
        # Save combined data back to file
        with open(qa_file_path, "w") as qa_file:
            json.dump(existing_data, qa_file, indent=4)
        st.sidebar.success("Danke für das Feedback!")

    # Function to save all Q&A data to QA.json file
    def save_all_qa_data():
        qa_file_path = os.path.join(current_dir, "db/QA.json")
        
        # Load existing data
        if os.path.exists(qa_file_path):
            with open(qa_file_path, "r") as qa_file:
                existing_data = json.load(qa_file)
        else:
            existing_data = []

        # Append new data
        existing_data.extend(st.session_state.qa_data)
        
        # Save combined data back to file
        with open(qa_file_path, "w") as qa_file:
            json.dump(existing_data, qa_file, indent=4)

    # Text input box for user input
    st.text_input("You: ", key="user_input", on_change=handle_user_input)

    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.write(f"You: {message.content}")
            elif isinstance(message, SystemMessage):
                st.write(f"HR AI: {message.content}")

    # Ask if the answer was helpful at the end of the conversation
    if st.session_state.chat_history and not st.session_state.feedback_given:
        st.write("War diese Antwort hilfreich?")
        col1, col2 = st.columns(2)
        if col1.button("Yes"):
            st.session_state.qa_data[-1]["correctAnswer"] = True
            st.session_state.feedback_given = True
            st.session_state.chat_id += 1  # Increment chat_id only once per session
            save_qa_data()
        if col2.button("No"):
            st.session_state.qa_data[-1]["correctAnswer"] = False
            st.session_state.feedback_given = True
            st.session_state.chat_id += 1  # Increment chat_id only once per session
            save_qa_data()

    # Sidebar for uploading PDFs
    st.sidebar.image('img/RH.png', width=250, clamp=False)
    with st.sidebar.expander("Add new Documents as Knowledge", expanded=True):
        uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                upload_directory = os.path.join(current_dir, "docs/uploaded_documents")
                os.makedirs(upload_directory, exist_ok=True)  # Ensure the uploaded_documents directory exists
                file_path = os.path.join(upload_directory, uploaded_file.name)

                if os.path.exists(file_path):
                    if st.sidebar.button(f"Overwrite {uploaded_file.name}?"):
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.sidebar.success(f"Overwritten {uploaded_file.name} successfully!")
                        convert_pdfs_in_folder(upload_directory)
                    else:
                        st.sidebar.warning(f"{uploaded_file.name} already exists.")
                else:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.sidebar.success(f"Uploaded {uploaded_file.name} successfully!")
                    convert_pdfs_in_folder(upload_directory)

    # Button to update the vector database
    if st.sidebar.button("Vektor-DB aktualisieren"):
        result = subprocess.run(["python", "tools/metadaten_to_DB.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Vektor-DB erfolgreich aktualisiert!")
        else:
            st.sidebar.error(f"Fehler beim Aktualisieren der Vektor-DB: {result.stderr}")
