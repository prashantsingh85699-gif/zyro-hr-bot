import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

# 1. Config & Setup
st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="centered")

# Env Variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
except:
    st.error("Secrets not configured correctly in Streamlit Cloud!")

# Pipeline Settings
FAISS_INDEX_PATH = "."
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 5
CROSS_ENCODER_THRESHOLD = 0.5
LLM_MODEL = "llama-3.3-70b-versatile"

# 2. Load Resources (Cached)
@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return vs, llm, rnk

# 3. RAG Pipeline
def get_answer(question):
    vs, llm, rnk = load_resources()
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": RETRIEVAL_K})
    docs = retriever.invoke(question)
    
    # Reranking Logic
    scores = rnk.predict([(question, d.page_content) for d in docs])
    final_docs = [d for s, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True) if s > CROSS_ENCODER_THRESHOLD]
    
    if not final_docs:
        return "I'm sorry, I couldn't find relevant policy information for this query.", []
    
    context = "\n\n".join([d.page_content for d in final_docs[:3]])
    sources = list(set([d.metadata.get("source", "Policy Doc") for d in final_docs]))
    
    prompt = ChatPromptTemplate.from_template("Answer based on Context: {context}\nQuestion: {question}")
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, sources

# 4. Streamlit UI
st.title("🏢 Zyro Dynamics HR Help Desk")
st.sidebar.info("Ask anything about Zyro HR Policies.")

# Question Logic
predefined = ["Select...", "What is the leave policy?", "How to claim insurance?", "Office hours?", "Holiday calendar?", "Remote work policy?"]
selected = st.selectbox("Quick Select:", predefined, key="sel")
user_input = st.chat_input("Or type your question...")

question = user_input if user_input else (selected if selected != "Select..." else None)

if question:
    with st.chat_message("user"): st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            ans, srcs = get_answer(question)
            st.markdown(f"**Answer:** {ans}")
            if srcs: st.caption(f"Sources: {', '.join(srcs)}")
