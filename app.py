import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

# 1. Page Config
st.set_page_config(page_title="Zyro HR Help Desk", layout="wide")

# 2. Key Handling (SABSE IMPORTANT: Agar Cloud par ho toh secrets, warna error)
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
except:
    st.error("⚠️ API Keys nahi mili! Streamlit Cloud Settings > Secrets mein keys add karo.")
    st.stop()

# 3. Resources (Cached)
@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.load_local(".", emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return vs, llm, rnk

vs, llm, rnk = load_resources()

# 4. Chat UI
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def process_query(question):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    docs = retriever.invoke(question)
    pairs = [(question, d.page_content) for d in docs]
    scores = rnk.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [d for s, d in ranked[:6] if s > 0.0]
    
    if not top_docs: return "No policy found.", []
    context = "\n\n".join(f"[{d.metadata.get('source', 'Policy')}]\n{d.page_content}" for d in top_docs)
    sources = list(set(d.metadata.get("source", "Policy Doc") for d in top_docs))
    
    chain = ChatPromptTemplate.from_template("Answer based on Context: {context}\nQuestion: {question}") | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}), sources

# 5. Input
question = st.chat_input("Ask HR policy question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        ans, srcs = process_query(question)
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
