import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

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

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

print("Chatbot ready! Type 'exit' to quit.\n")
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        break
    answer = chain.invoke(question)
    print(f"\nBot: {answer.content}\n")