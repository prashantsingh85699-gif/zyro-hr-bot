# Updated app.py with 5+ Quick Select questions
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

# Setup Environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🧑‍💼")
st.title("🧑‍💼 Zyro Dynamics HR Help Desk")

# Pipeline Configuration
FAISS_INDEX_PATH = "." 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 5
FINAL_TOP_K = 3
CROSS_ENCODER_THRESHOLD = 0.5
LLM_MODEL = "llama-3.3-70b-versatile"
REFUSAL_MESSAGE = "The HR policy documents do not contain information regarding this query. Please contact HR."

# Prompt Template
PROMPT = ChatPromptTemplate.from_template(
    "You are an expert HR assistant for Zyro Dynamics.\n"
    "Answer using ONLY the provided context. If nothing relevant is found, say: " + REFUSAL_MESSAGE + "\n\n"
    "Context: {context}\nQuestion: {question}\n\nDetailed Answer:"
)

@st.cache_resource
def load_resources():
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=1536)
    rnk = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return vs, llm, rnk

def answer_question(vs, llm, rnk, question):
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": RETRIEVAL_K, "fetch_k": 10})
    raw_docs = retriever.invoke(question)
    if not raw_docs: return REFUSAL_MESSAGE, []
    
    pairs = [(question, d.page_content) for d in raw_docs]
    scores = rnk.predict(pairs)
    if max(scores) < CROSS_ENCODER_THRESHOLD: return REFUSAL_MESSAGE, []
    
    ranked = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
    final = [d for _, d in ranked][:FINAL_TOP_K]
    
    context = "\n\n".join(f"[{d.metadata.get('source','?')}] {d.page_content}" for d in final)
    sources = sorted({d.metadata.get("source", "unknown") for d in final})
    answer = (PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, sources

vs, llm, rnk = load_resources()

# Updated Quick Select Logic (5+ questions)
predefined_questions = [
    "Select a question...",
    "What is the company leave policy?",
    "How do I claim health insurance?",
    "What are the office working hours?",
    "Where can I find the holiday calendar?",
    "What is the remote work policy?",
    "How do I submit an expense report?"
]

selected_q = st.selectbox("Quick Select:", predefined_questions)
chat_input = st.chat_input("Or type your HR policy question here...")

question = selected_q if selected_q != "Select a question..." else chat_input

if question:
    with st.chat_message("user"): st.write(question)
    with st.chat_message("assistant"):
        answer, sources = answer_question(vs, llm, rnk, question)
        st.markdown(f"**Answer:** {answer}")
        if sources: st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in sources))
