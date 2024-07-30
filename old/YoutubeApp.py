import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

#App framework
st.image('img/RH.png', use_column_width=True)
st.title("Rheinmetall HR Bot")
# Abfrage des Namens
name = st.text_input("Bitte geben Sie Ihren Namen/CA-Nummer ein:")

prompt = st.text_input("Wie kann ich Ihnen helfen ?")

#Prompt templates
titel_template = PromptTemplate(
    input_variables= ['topic'],
    template="Antworte immer auf Deutsch!. Write me a YouTube video title about {topic}"
)
script_template = PromptTemplate(
    input_variables= ['title', 'wikipedia_research'],
    template="Write me a YouTube script based on this title TITLE: {title} while leveraging this wikipedia research: {wikipedia_research} "
)

# Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# LLMs
llm = OpenAI(temperature=0.9, model="gpt-3.5-turbo-instruct") #gpt-3.5-turbo, text-embedding-ada-002
title_chain = LLMChain(llm=llm, prompt=titel_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
#sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

wiki = WikipediaAPIWrapper()

#show response frpm OAI
if prompt:
    #response = sequential_chain({'topic':prompt})
    title= title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(f"Hallo {name}, hier ist die Antwort auf Ihre Anfrage:")
    st.write(title) 
    st.write(script)

    with st.expander("Title history"):
        st.info(title_memory.buffer)

    with st.expander("Script history"):
        st.info(script_memory.buffer)

    with st.expander("Wiki history"):
        st.info(wiki_research)