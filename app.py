import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# 1. UI Setup
st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="wide")

# 2. Secret Handling (Fool-proof)
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ API Key nahi mili! Streamlit Cloud Settings > Secrets mein GROQ_API_KEY add karo.")
    st.stop()
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# 3. Sidebar
with st.sidebar:
    st.title("🏢 Zyro Dynamics")
    st.markdown("### HR Policy AI Assistant")
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# 4. Resources (Fixed Path)
@st.cache_resource
def load_resources():
    # . dhoondta hai current root directory mein
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(".", emb, allow_dangerous_deserialization=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return vs, llm, rnk

vs, llm, rnk = load_resources()

# 5. UI Logic
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Sources"): st.write(", ".join(msg["sources"]))

# 6. Chat Logic
predefined = ["Select...", "What is the leave policy?", "How to claim insurance?"]
selected = st.selectbox("Quick Select:", predefined)
user_input = st.chat_input("Type your question here...")
question = user_input if user_input else (selected if selected != "Select..." else None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing policy..."):
            # Retrieval
            docs = vs.similarity_search(question, k=5)
            pairs = [(question, d.page_content) for d in docs]
            scores = rnk.predict(pairs)
            final_docs = [d for s, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True) if s > 0.5]
            
            if not final_docs:
                ans = "Sorry, no relevant policy found."
                srcs = []
            else:
                context = "\n\n".join([d.page_content for d in final_docs[:3]])
                srcs = list(set([d.metadata.get("source", "Policy Doc") for d in final_docs]))
                prompt = ChatPromptTemplate.from_template("Answer based on Context: {context}\nQuestion: {question}")
                ans = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
            
            st.markdown(ans)
            if srcs:
                with st.expander("View Sources"): st.write(", ".join(srcs))
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
