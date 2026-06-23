import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. UI Configuration
st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="wide")

# 2. Secret Handling
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is not set in Streamlit Secrets!")
    st.stop()
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# 3. Resource Loading (Fixed Path)
@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    # Ye line current folder se files load karegi
    vs = FAISS.load_local(os.getcwd(), emb, allow_dangerous_deserialization=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    return vs, llm

vs, llm = load_resources()

# 4. Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. UI Layout
st.title("🏢 Zyro Dynamics HR Help Desk")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Logic
user_input = st.chat_input("Ask your HR policy question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            # Retrieval
            docs = vs.similarity_search(user_input, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            # Response
            prompt = ChatPromptTemplate.from_template("Answer based on context: {context}\nQuestion: {question}")
            chain = prompt | llm | StrOutputParser()
            ans = chain.invoke({"context": context, "question": user_input})
            
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
