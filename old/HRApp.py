import os
from apikey import apikey
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import openai


# Setze deinen OpenAI API-Schlüssel
openai.api_key  = apikey

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to process a CSV file
def load_csv(csv_file):
    return pd.read_csv(csv_file)

# App framework
st.image('img/RH.png', width=500) #use_column_width=True,
st.title("Rheinmetall HR Bot 2000")

# Prompt for user name
name = st.text_input("Bitte geben Sie Ihren Namen/CA-Nummer ein:")

# File uploads
uploaded_pdf = st.file_uploader("Laden Sie eine PDF-Datei hoch", type="pdf")
uploaded_csv = st.file_uploader("Laden Sie eine CSV-Datei hoch", type="csv")

# User query
query = st.text_input("Geben Sie Ihre Anfrage ein:")

# Process uploaded files and user query
if uploaded_pdf or uploaded_csv:
    # Process the PDF file
    if uploaded_pdf:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        #st.write("Text aus der PDF-Datei extrahiert:")
        #spst.write(pdf_text)

    # Process the CSV file
    if uploaded_csv:
        csv_data = load_csv(uploaded_csv)
        st.write("Daten aus der CSV-Datei geladen:")
        st.write(csv_data)

    # Use OpenAI to search for information
    if query:
        # Combine data from both files
        combined_data = ""
        if uploaded_pdf:
            combined_data += pdf_text + "\n\n"
        if uploaded_csv:
            combined_data += csv_data.to_string() + "\n\n"

        # Query OpenAI
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            temperature=0.1,
            prompt=f"Mein Name ist {name}. Antworte immer in der Sie Form. Beantworte die folgende Frage basierend auf diesen Daten:\n\n{combined_data}\n\nFrage: {query}",
            max_tokens=300
        )

        st.write(f"Hallo {name}:")
        st.write(response.choices[0].text.strip())