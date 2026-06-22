import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever  # ✅ NEW
from langchain_community.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder              # ✅ NEW

# Setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# API Keys from Streamlit Secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🧑‍💼")
st.title("🧑‍💼 Zyro Dynamics HR Help Desk")

# ✅ UPDATED THRESHOLD (was 0.65, now 1.5 to match notebook)
RELEVANCE_THRESHOLD = 1.5
REFUSAL_MESSAGE = "I don't have enough information in the HR policy documents to answer this question."
FAISS_INDEX_PATH = "."
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ BETTER PROMPT (same as notebook)
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are an expert HR assistant for Zyro Dynamics company.
Use ONLY the context below to answer the question accurately and completely.
If the answer is not in the context, say "The provided policy documents do not contain information regarding this query."

Context: {context}
Question: {question}

Provide a detailed, accurate answer based strictly on the context:"""
)

@st.cache_resource
def load_resources():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # ✅ NEW
    return vectorstore, llm, reranker

def rerank_results(reranker, query, docs):
    """✅ NEW — rerank docs by relevance score"""
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = [doc for doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked[:5]  # top 5

def answer_question(vectorstore, llm, reranker, question: str):
    # Guardrail check
    results = vectorstore.similarity_search_with_score(question, k=1)
    top_score = results[0][1] if results else 1.5
    
    # ✅ UPDATED: score < 1.5 (was < 0.65)
    if top_score >= RELEVANCE_THRESHOLD:
        return REFUSAL_MESSAGE, [], top_score

    # ✅ NEW: BM25 + FAISS Ensemble Retriever
    # Load all docs from vectorstore for BM25
    all_docs = vectorstore.similarity_search("", k=1000)  
    
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5

    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.4},  # ✅ UPDATED
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

    # Retrieve + Rerank
    raw_docs = hybrid_retriever.invoke(question)
    final_docs = rerank_results(reranker, question, raw_docs)  # ✅ NEW

    context = "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" 
        for d in final_docs
    )
    sources = sorted({d.metadata.get("source", "unknown") for d in final_docs})

    chain = PROMPT_TEMPLATE | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, sources, top_score

# Main App
vectorstore, llm, reranker = load_resources()  # ✅ UPDATED (added reranker)

# Callback to reset selectbox
def clear_select():
    st.session_state.selected_q = "Select a question..."

# UI: Dropdown + Chat Input
predefined_questions = [
    "Select a question...",
    "What is the company leave policy?",
    "How do I claim health insurance?",
    "What are the office working hours?",
    "Where can I find the holiday calendar?"
]

selected_question = st.selectbox(
    "Quick Select:",
    predefined_questions,
    key="selected_q"
)
chat_input = st.chat_input("Or type your HR policy question here...", on_submit=clear_select)

# Determine active question
question = None
if selected_question != "Select a question...":
    question = selected_question
elif chat_input:
    question = chat_input

# Processing
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, sources, score = answer_question(vectorstore, llm, reranker, question)  # ✅ UPDATED
            st.markdown(f"**Answer:** {answer}")
            if sources:
                st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in sources))
