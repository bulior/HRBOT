import os
from apikey import apikey

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import openai

# Set your OpenAI API key
openai.api_key = apikey
os.environ['OPENAI_API_KEY'] = apikey

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to extract text from a PDF file path
def extract_text_from_pdf_path(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to process a CSV file
def load_csv(csv_file):
    return pd.read_csv(csv_file)

# App framework
st.image('img/RH.png', width=500) #use_column_width=True,
st.title("Rheinmetall HR Bot")

# Prompt for user name
name = st.text_input("Bitte geben Sie Ihren Namen/CA-Nummer ein:")

# File uploads
uploaded_pdf = 0#st.file_uploader("Laden Sie eine PDF-Datei hoch", type="pdf")
uploaded_csv = st.file_uploader("Laden Sie eine CSV-Datei hoch", type="csv")

# User query
query = st.text_input("Wie kann Ich Ihnen helfen:")

# Process uploaded files and user query
if uploaded_pdf or uploaded_csv or os.path.exists("docs"):
    combined_data = ""

    # Process the PDF file
    #if uploaded_pdf:
    #    pdf_text = extract_text_from_pdf(uploaded_pdf)
    #    combined_data += pdf_text + "\n\n"
        #st.write("Text aus der PDF-Datei extrahiert:")
        #st.write(pdf_text)

    # Process the CSV file
    if uploaded_csv:
        csv_data = load_csv(uploaded_csv)
        combined_data += csv_data.to_string() + "\n\n"
        st.write("Daten aus der CSV-Datei geladen:")
        st.write(csv_data)

    # Process PDFs in the "docs" folder
    if os.path.exists("docs"):
        #st.write("Gefundene PDF-Dateien im Ordner 'docs':")
        for filename in os.listdir("docs"):
            if filename.endswith(".pdf"):
                #pdf_path = os.path.join("docs", filename)
                #st.write(f"- {filename}")
                #pdf_text = extract_text_from_pdf_path(pdf_path)
                #combined_data += pdf_text + "\n\n"
                pass

    # Use OpenAI API to manage large texts
    if query:
        # Split the combined data into manageable chunks
        def split_text(text, chunk_size=1000, chunk_overlap=50):
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunks.append(text[i:i + chunk_size])
            return chunks

        texts = split_text(combined_data)

        # Define the prompt template
        prompt_template = ("Du bist eine HR Bot der Fragen beantwortet und sich die Information aus den zur verfügung gestellten unterlagen sucht."
                           "Antworte immer in der Sie Form und in der Sprache wie gefragt wird. "
                           "Gebe bei Fragen immer deine Quelle an, welche Datei benutzt wurde und die Seitenzahl der gefunden Information. "
                           "Antworte so kurz wie möglich!"
                           "Falls du keine geneu Quelle hast, sage dass du nicht genug Informationen für die Frage besitzt. "
                           "Antworte kurz und informatiev und dichte nichts hinzu. "
                           "Beantworte die folgende Frage basierend auf diesen Daten:\n\n{text}\n\nFrage: {query}")

        # Collect responses
        responses = []
        for text in texts:
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct",
                temperature=0,
                prompt=prompt_template.format(text=text, query=query),
                max_tokens=100
            )
            responses.append(response.choices[0].text.strip())

        # Display responses
        st.write(f"Hallo {name}:")
        for response in responses:
            st.write(response)

