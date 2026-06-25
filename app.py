# import os
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores.utils import DistanceStrategy
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
# from sentence_transformers import CrossEncoder

# # Page Configuration
# st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢", layout="wide")

# # Setup Environment
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
# os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# # Sidebar Branding
# with st.sidebar:
#     st.title("🏢 Zyro Dynamics")
#     st.markdown("### HR Policy AI Assistant")
#     st.info("Ask me anything about company policies.")

# @st.cache_resource
# def load_resources():
#     emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
#     # Path is set to "." as index files are in the root directory
#     vs = FAISS.load_local(".", emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
#     llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
#     rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     return vs, llm, rnk

# vs, llm, rnk = load_resources()

# # Session State for History
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display Chat
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if "sources" in msg and msg["sources"]:
#             with st.expander("View Sources"):
#                 st.write(", ".join(msg["sources"]))

# # RAG Pipeline
# def process_query(question):
#     retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5})
#     docs = retriever.invoke(question)
#     pairs = [(question, d.page_content) for d in docs]
#     scores = rnk.predict(pairs)
    
#     final_docs = [d for s, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True) if s > 0.5]
#     if not final_docs: return "Sorry, no relevant policy found.", []
    
#     context = "\\n\\n".join([d.page_content for d in final_docs[:3]])
#     sources = list(set([d.metadata.get("source", "Policy Doc") for d in final_docs]))
    
#     prompt = ChatPromptTemplate.from_template("Answer based on Context: {context}\\nQuestion: {question}")
#     answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
#     return answer, sources

# # UI Input
# predefined = ["Select...", "What is the leave policy?", "How to claim insurance?", "Office hours?", "Holiday calendar?"]
# selected = st.selectbox("Quick Select:", predefined)
# user_input = st.chat_input("Type your question here...")

# question = user_input if user_input else (selected if selected != "Select..." else None)

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"): st.markdown(question)
    
#     with st.chat_message("assistant"):
#         with st.spinner("Analyzing policy..."):
#             ans, srcs = process_query(question)
#             st.markdown(ans)
#             if srcs:
#                 with st.expander("View Sources"):
#                     st.write(", ".join(srcs))
#             st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
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
st.set_page_config(page_title="Zyro HR Compliance Auditor", page_icon="🏢", layout="wide")

# 2. Environment Setup (Ensure secrets are set in Streamlit Cloud)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# 3. Zenith Prompt Template
prompt_template = """
SYSTEM: You are the Zyro Dynamics HR Compliance Auditor. Your task is to perform high-precision information extraction.

### MANDATORY RULES:
1. **AUTHORITY**: Answer strictly using the provided Context. If the information is missing, output exactly: "The HR policy documents do not contain information regarding this query."
2. **CITATION**: Every claim must end with [Source: filename.pdf].
3. **ZERO CONVERSATIONAL FILLER**: No intros/outros. Start immediately with the answer.
4. **VERBATIM**: Quote figures/dates exactly as written. No rounding.

### OUTPUT FORMAT:
- Steps: Numbered list (1., 2., ...).
- Rules/Benefits: Bullet points.
- Facts: Direct, concise sentence.

---
CONTEXT: {context}

QUESTION: {question}

ANSWER:
"""

# 4. Resource Loading (Cached for Speed)
@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
    # Make sure index files are in the same folder as app.py
    vs = FAISS.load_local(".", emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return vs, llm, rnk

vs, llm, rnk = load_resources()

# 5. UI - Sidebar & Chat History
with st.sidebar:
    st.title("🏢 Zyro HR Auditor")
    st.markdown("### Compliance AI Assistant")
    st.info("Ask about company policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                st.write(", ".join(msg["sources"]))

# 6. RAG Pipeline with Relevance Guardrails
def process_query(question):
    # Retrieve
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    docs = retriever.invoke(question)
    
    # Rerank
    pairs = [(question, d.page_content) for d in docs]
    scores = rnk.predict(pairs)
    
    # Relevance Thresholding (The Guardrail)
    final_docs = [d for s, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True) if s > 0.1]
    
    # Handle Refusal
    if not final_docs: 
        return "The HR policy documents do not contain information regarding this query.", []
    
    # Context Construction
    context = "\n\n".join([d.page_content for d in final_docs[:5]])
    
    # Source Cleaning (basename ensures clean output)
    sources = list({os.path.basename(d.metadata.get("source", "unknown.pdf")) for d in final_docs})
    
    # Execution
    prompt = ChatPromptTemplate.from_template(prompt_template)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, sources

# 7. UI Input Handling
question = st.chat_input("Enter your HR policy query...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Auditing..."):
            ans, srcs = process_query(question)
            st.markdown(ans)
            if srcs:
                with st.expander("View Sources"):
                    st.write(", ".join(srcs))
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
