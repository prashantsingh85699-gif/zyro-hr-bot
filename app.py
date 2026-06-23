import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

# Page Configuration
st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="wide")

# Sidebar Branding
with st.sidebar:
    st.title("🏢 Zyro Dynamics")
    st.markdown("### HR Policy AI Assistant")
    st.info("Ask me anything about company leave, insurance, or office policies.")
    
# Initialize Resources
@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return vs, llm, rnk

vs, llm, rnk = load_resources()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                st.write(", ".join(msg["sources"]))

# Pipeline
def process_query(question):
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    pairs = [(question, d.page_content) for d in docs]
    scores = rnk.predict(pairs)
    
    final_docs = [d for s, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True) if s > 0.5]
    if not final_docs: return "Sorry, no policy found.", []
    
    context = "\n\n".join([d.page_content for d in final_docs[:3]])
    sources = list(set([d.metadata.get("source", "Policy Doc") for d in final_docs]))
    
    prompt = ChatPromptTemplate.from_template("Answer based on Context: {context}\nQuestion: {question}")
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, sources

# Input Logic
predefined = ["Select...", "What is the leave policy?", "How to claim insurance?", "Office hours?", "Holiday calendar?"]
selected = st.selectbox("Quick Select:", predefined)
user_input = st.chat_input("Type your question here...")

question = user_input if user_input else (selected if selected != "Select..." else None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing policy..."):
            ans, srcs = process_query(question)
            st.markdown(ans)
            if srcs:
                with st.expander("View Sources"):
                    st.write(", ".join(srcs))
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
