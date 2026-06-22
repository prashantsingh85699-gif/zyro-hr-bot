import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🧑‍💼")
st.title("🧑‍💼 Zyro Dynamics HR Help Desk")

RELEVANCE_THRESHOLD = 1.5
REFUSAL_MESSAGE = "I don't have enough information in the HR policy documents to answer this question."
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are an expert HR assistant for Zyro Dynamics company.
Use ONLY the context below to answer completely and accurately.
Include specific numbers, days, and percentages when present.
If not in context say: "The HR policy documents do not contain information regarding this query."

Context: {context}
Question: {question}

Detailed Answer:"""
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
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=1024)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return vectorstore, llm, reranker

def rerank_results(reranker, query, docs):
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked[:5]

def answer_question(vectorstore, llm, reranker, question):
    results = vectorstore.similarity_search_with_score(question, k=1)
    top_score = results[0][1] if results else 1.5
    if top_score >= RELEVANCE_THRESHOLD:
        return REFUSAL_MESSAGE, [], top_score
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.4},
    )
    raw_docs = retriever.invoke(question)
    final_docs = rerank_results(reranker, question, raw_docs)
    context = "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}] {d.page_content}"
        for d in final_docs
    )
    sources = sorted({d.metadata.get("source", "unknown") for d in final_docs})
    chain = PROMPT_TEMPLATE | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, sources, top_score

vectorstore, llm, reranker = load_resources()

def clear_select():
    st.session_state.selected_q = "Select a question..."

predefined_questions = [
    "Select a question...",
    "What is the company leave policy?",
    "How do I claim health insurance?",
    "What are the office working hours?",
    "Where can I find the holiday calendar?"
]

selected_question = st.selectbox("Quick Select:", predefined_questions, key="selected_q")
chat_input = st.chat_input("Or type your HR policy question here...", on_submit=clear_select)

question = None
if selected_question != "Select a question...":
    question = selected_question
elif chat_input:
    question = chat_input

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, sources, score = answer_question(vectorstore, llm, reranker, question)
            st.markdown(f"**Answer:** {answer}")
            if sources:
                st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in sources))
