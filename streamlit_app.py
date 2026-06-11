import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="M3M Altitude Chatbot", page_icon="🏢")
st.title("🏢 M3M Altitude — Ask Anything")
st.caption("AI powered by your property brochure")

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_template("""
    Answer based on context only:
    {context}
    Question: {question}
    """)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about the property..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(prompt).content
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})