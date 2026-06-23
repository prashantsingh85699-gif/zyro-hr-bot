app_template = '''
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="wide")

# --- PRO ENVIRONMENT SETUP ---
def setup_env():
    # Try fetching from Streamlit secrets first, then from system environment variables
    try:
        os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", os.environ.get("LANGCHAIN_API_KEY", ""))
    except Exception:
        pass # Handle case where secrets might not be loaded yet

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"

setup_env()

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("🏢 Zyro Dynamics")
    st.markdown("### HR Policy AI Assistant")
    st.info("Ask me anything about company HR policies.")
    st.caption("🚀 Powered by BGE-Large + LLaMA 3.3 70B")

# --- RESOURCE LOADING (Optimized) ---
@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", encode_kwargs={"normalize_embeddings": True})
    # Loading index from root
    vs  = FAISS.load_local(".", emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=2048)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return vs, llm, rnk

vs, llm, rnk = load_resources()

# --- CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Source Documents"):
                st.write(", ".join(msg["sources"]))

# --- RAG PIPELINE ---
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert HR policy assistant for Zyro Dynamics. Answer using ONLY the provided context. 
Be concise, use bullet points, and address all parts of the user question.

Context: {context}
Question: {question}
Answer:""")

def process_query(question):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    docs      = retriever.invoke(question)
    pairs     = [(question, d.page_content) for d in docs]
    scores    = rnk.predict(pairs)
    ranked    = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs  = [d for s, d in ranked[:6] if s > 0.0]
    
    if not top_docs:
        return "I'm sorry, I couldn't find a policy for that specific query.", []
        
    context = "\\n\\n".join(f"[{d.metadata.get('source', 'Policy')}]\\n{d.page_content}" for d in top_docs)
    sources = list(set(d.metadata.get("source", "Policy Doc") for d in top_docs))
    
    chain   = RAG_PROMPT | llm | StrOutputParser()
    answer  = chain.invoke({"context": context, "question": question})
    return answer, sources

# --- UI INPUT ---
predefined = ["Select a sample question...", "What is the leave policy?", "How do I claim health insurance?", "What are office working hours?", "What is the work from home policy?"]
selected = st.selectbox("Quick questions:", predefined)
user_input = st.chat_input("Type your HR question here...")
question = user_input if user_input else (selected if selected != predefined[0] else None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing HR docs..."):
            ans, srcs = process_query(question)
            st.markdown(ans)
            if srcs:
                with st.expander("View Source Documents"):
                    st.write(", ".join(srcs))
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
'''

with open("app.py", "w") as f:
    f.write(app_template.strip())

print("✅ Pro-version app.py generated successfully!")
