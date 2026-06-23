import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# 1. UI Setup
st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="wide")

# 2. Secret Handling (Fool-proof)
def get_key(key_name):
    # Try streamlit secrets first
    if key_name in st.secrets:
        return st.secrets[key_name]
    # Then try environment variables
    return os.environ.get(key_name)

GROQ_KEY = get_key("GROQ_API_KEY")
if not GROQ_KEY:
    st.error("❌ API Key nahi mili! Streamlit Cloud Secrets mein GROQ_API_KEY daalo.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_KEY

# 3. Sidebar
with st.sidebar:
    st.title("🏢 Zyro Dynamics")
    st.markdown("### HR Policy AI Assistant")
    st.info("Ask me anything about company HR policies.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# 4. Resources
@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vs = FAISS.load_local(".", emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return vs, llm, rnk

vs, llm, rnk = load_resources()

# 5. UI Logic
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 6. Chat Input
predefined = ["Select...", "What is the leave policy?", "How to claim health insurance?"]
selected = st.selectbox("Quick questions:", predefined)
user_input = st.chat_input("Type your question...")
question = user_input if user_input else (selected if selected != predefined[0] else None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policy..."):
            # Retrieval
            docs = vs.similarity_search(question, k=10)
            pairs = [(question, d.page_content) for d in docs]
            scores = rnk.predict(pairs)
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            top_docs = [d for s, d in ranked[:3]]
            
            context = "\n".join([d.page_content for d in top_docs])
            prompt = ChatPromptTemplate.from_template("Answer using this context: {context}\nQuestion: {question}")
            chain = prompt | llm | StrOutputParser()
            ans = chain.invoke({"context": context, "question": question})
            
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
