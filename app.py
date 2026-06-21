import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# API Keys from Streamlit Secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🧑‍💼")
st.title("🧑‍💼 Zyro Dynamics HR Help Desk")

RELEVANCE_THRESHOLD = 0.65
REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."
FAISS_INDEX_PATH = "." 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are the Zyro Dynamics HR expert. 
    1. Use ONLY the provided context. 
    2. If the user asks about HR/Policy, answer using the context. 
    3. If the answer is not in the context, explicitly state: 'The provided policy documents do not contain information regarding this query.'
    
    Context: {context}
    Question: {question}
    """
)

@st.cache_resource
def load_resources():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return vectorstore, llm

def answer_question(vectorstore, llm, question: str):
    results = vectorstore.similarity_search_with_score(question, k=3) 
    top_score = results[0][1] if results else 0.0

    if top_score < RELEVANCE_THRESHOLD:
        return REFUSAL_MESSAGE, [], top_score

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs)
    sources = sorted({d.metadata.get("source", "unknown") for d in docs})

    chain = PROMPT_TEMPLATE | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, sources, top_score

# Main App logic
vectorstore, llm = load_resources()

# Callback to reset selectbox when chat_input is used
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

# Determine the active question
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
            answer, sources, score = answer_question(vectorstore, llm, question)
            st.markdown(f"**Answer:** {answer}")
            if sources:
                st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in sources))
