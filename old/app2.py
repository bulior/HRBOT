import os
import datetime
import json
import getpass
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document  # Importiere die Document-Klasse

# Load environment variables from .env
load_dotenv()

# Define the persistent directory for the main database
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the paths for the QA JSON file and the Questions JSON file
qa_json_path = os.path.join(current_dir, "db/QA.json")
questions_json_path = os.path.join(current_dir, "db/Q.json")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load the existing vector store with the embedding function
try:
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    print("Debug: Hauptdatenbank erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden der Hauptdatenbank: {e}")

# Create a ChatOpenAI model
try:
    model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    print("Debug: ChatOpenAI Modell erfolgreich erstellt.")
except Exception as e:
    print(f"Fehler beim Erstellen des ChatOpenAI Modells: {e}")

# Get the current PC username
pc_username = getpass.getuser()

# Streamlit app
st.image('img/RH.png', width=200)
st.title("Wie kann ich Ihnen helfen?")

query = st.text_input("Stellen Sie eine Frage:")

if 'result' not in st.session_state:
    st.session_state.result = None
if 'document' not in st.session_state:
    st.session_state.document = None

if st.button("Frage stellen"):
    if query:
        with st.spinner("Bitte warten Sie, die Anfrage wird bearbeitet..."):
            try:
                # Save the question to Q.json
                timestamp = str(datetime.datetime.now())
                question_entry = {
                    'date': timestamp,
                    'question': query,
                    'user': pc_username
                }

                if os.path.exists(questions_json_path):
                    with open(questions_json_path, 'r') as file:
                        questions_data = json.load(file)
                else:
                    questions_data = []

                questions_data.append(question_entry)

                with open(questions_json_path, 'w') as file:
                    json.dump(questions_data, file, indent=4)

                # Retrieve relevant documents based on the query
                retriever = db.as_retriever(search_type="mmr", lambda_mult=1)
                relevant_docs = retriever.invoke(query)
                print(f"Debug: relevante Dokumente abgerufen: {relevant_docs}")

                # Display the relevant results with metadata
                st.markdown('<p style="color:lightblue;">\nFolgendes konnte ich in meinen Dokumenten finden:</p>', unsafe_allow_html=True)
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f'<p style="color:gray;">Document {i} Title:\n{doc.metadata.get("title", "No Title")}</p>', unsafe_allow_html=True)
                    print(f"Debug: Dokument {i} - Titel: {doc.metadata.get('title', 'No Title')}")

                # Combine the query and the relevant document contents
                combined_input = (
                    "Hier sind Dokumente, die nur deine Datenbasis bilden. Suche jedes Dokument sehr sorgfältig ab und dicht zu den Antworten nichts hinzu! Wenn du mehrer Antworten zu einer Frage findest, Gruppiere nach den jeweiligen Dokumenten, in denen du etwas gefunden hast. Gebe möglichst ein zu eins aus den Dokumenten wieder. "
                    + query
                    + "\n\nRelevant Documents:\n"
                    + "\n\n".join([doc.page_content for doc in relevant_docs])
                    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with: 'Ich bin mir nicht sicher, bitte wenden Sie sich an die zuständige Abteilung'."
                )
                print(f"Debug: Kombinierte Eingabe für das Modell: {combined_input}")

                # Define the messages for the model
                messages = [
                    SystemMessage(content="Du bist eine Hilfreicher Antwortenbot für jegliche Fragen. Antworte immer in der Sprache wie die Frage gestellt wurde."),
                    HumanMessage(content=combined_input),
                ]

                # Invoke the model with the combined input
                result = model.invoke(messages)
                print(f"Debug: Ergebnis vom Modell: {result}")

                # Display the full result and content only
                st.session_state.result = result.content
                st.write(result.content)

                # Create document for feedback storage
                st.session_state.document = Document(
                    page_content=query + "\n" + result.content,
                    metadata={
                        'date': timestamp,
                        'correctAnswer': None,
                        'question': query,
                        'answer': result.content,
                        'user': pc_username
                    }
                )

            except Exception as e:
                st.write(f"Fehler bei der Bearbeitung der Anfrage: {e}")
                print(f"Fehler bei der Bearbeitung der Anfrage: {e}")
    else:
        st.write("Bitte geben Sie eine Frage ein.")

def save_feedback_to_json(document):
    feedback_entry = {
        'date': document.metadata['date'],
        'question': document.metadata['question'],
        'answer': document.metadata['answer'],
        'correctAnswer': document.metadata['correctAnswer'],
        'user': document.metadata['user']
    }

    if os.path.exists(qa_json_path):
        with open(qa_json_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    data.append(feedback_entry)

    with open(qa_json_path, 'w') as file:
        json.dump(data, file, indent=4)

if st.session_state.result:
    st.markdown("<p style='color:lightblue;'>War die Antwort hilfreich?</p>", unsafe_allow_html=True)
    if st.button("Ja"):
        try:
            st.session_state.document.metadata['correctAnswer'] = True
            save_feedback_to_json(st.session_state.document)
            st.write("Danke für Ihr Feedback!")
            print('Feedback gut: Ja')
        except Exception as e:
            st.write(f"Fehler beim Speichern des Feedbacks: {e}")
            print(f"Fehler beim Speichern des Feedbacks (Ja): {e}")

    if st.button("Nein"):
        try:
            st.session_state.document.metadata['correctAnswer'] = False
            save_feedback_to_json(st.session_state.document)
            st.write("Ihre Frage und Antwort wurden gespeichert. Danke für Ihr Feedback!")
            print('Feedback gut: Nein')
        except Exception as e:
            st.write(f"Fehler beim Speichern des Feedbacks: {e}")
            print(f"Fehler beim Speichern des Feedbacks (Nein): {e}")

