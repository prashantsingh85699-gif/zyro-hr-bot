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

# 2. Secret Handling
def get_key(key_name):
    if key_name in st.secrets:
        return st.secrets[key_name]
    return os.environ.get(key_name)

GROQ_KEY = get_key("GROQ_API_KEY")
if not GROQ_KEY:
    st.error("❌ API Key not found in Streamlit Secrets!")
    st.stop()
os.environ["GROQ_API_KEY"] = GROQ_KEY

# 3. Resources (Path Fix)
@st.cache_resource
def load_resources():
    # 'os.getcwd()' current directory ka absolute path deta hai
    current_dir = os.getcwd() 
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    # Check if files exist at root
    if not os.path.exists(os.path.join(current_dir, "index.faiss")):
        st.error(f"❌ index.faiss file nahi mili! Current directory: {current_dir}")
        st.stop()
        
    vs = FAISS.load_local(current_dir, emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return vs, llm, rnk

vs, llm, rnk = load_resources()

# 4. Chat Interface
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 5. Chat Input & Processing
user_input = st.chat_input("Type your HR question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            docs = vs.similarity_search(user_input, k=10)
            pairs = [(user_input, d.page_content) for d in docs]
            scores = rnk.predict(pairs)
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            top_docs = [d for s, d in ranked[:3]]
            
            context = "\n".join([d.page_content for d in top_docs])
            prompt = ChatPromptTemplate.from_template("Answer using this context: {context}\nQuestion: {question}")
            chain = prompt | llm | StrOutputParser()
            ans = chain.invoke({"context": context, "question": user_input})
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
