import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

# 1. Page Configuration
st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="wide")

# 2. Environment Setup (Notebook & Cloud Compatible)
def setup_env():
    try:
        os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", os.environ.get("LANGCHAIN_API_KEY", ""))
    except: pass
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

setup_env()

# 3. Sidebar UI (Branding & Controls)
with st.sidebar:
    st.title("🏢 Zyro Dynamics")
    st.markdown("### HR Policy AI Assistant")
    st.info("Ask me anything about company HR policies.")
    st.caption("Powered by BAAI/bge-large + LLaMA 3.3 70B")
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# 4. Resource Loader
@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.load_local(".", emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=2048)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return vs, llm, rnk

vs, llm, rnk = load_resources()

# 5. Chat Logic & History
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Source Documents"):
                st.write(", ".join(msg["sources"]))

# 6. RAG Pipeline
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert HR policy assistant for Zyro Dynamics. Answer using ONLY the context. 
Copy numbers, dates, and policy codes exactly. Use bullet points for lists.

Context: {context}
Question: {question}
Answer:""")

def process_query(question):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    docs = retriever.invoke(question)
    pairs = [(question, d.page_content) for d in docs]
    scores = rnk.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [d for s, d in ranked[:6] if s > 0.0]
    if not top_docs: return "No relevant policy found.", []
    context = "\n\n".join(f"[{d.metadata.get('source', 'Policy')}]\n{d.page_content}" for d in top_docs)
    sources = list(set(d.metadata.get("source", "Policy Doc") for d in top_docs))
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}), sources

# 7. UI Input (Dropdown + Chat)
predefined = ["Select a sample question...", "What is the leave policy?", 
              "How do I claim health insurance?", "What are office working hours?",
              "What is the work from home policy?", "How is performance reviewed?"]
selected = st.selectbox("Quick questions:", predefined)
user_input = st.chat_input("Type your HR question here...")
question = user_input if user_input else (selected if selected != predefined[0] else None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policy documents..."):
            ans, srcs = process_query(question)
            st.markdown(ans)
            if srcs:
                with st.expander("View Source Documents"): st.write(", ".join(srcs))
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
