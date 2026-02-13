import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import re
import os
groq_api_key= os.getenv("groq_api_key")


if not groq_api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()

# Setup models
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.0
)

Settings.embed_model = embed_model
Settings.llm = llm

# Load documents
documents = SimpleDirectoryReader(input_dir="Data").load_data()

for doc in documents:
    doc.metadata = {"college": "SVERI"}


index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=8)

def detect_greeting(text):
    text = text.lower().strip()

    # Normalize repeated letters (hiiii → hii)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    if re.search(r"\bgood morning\b", text):
        return "Good Morning ☀ How can I assist you with admissions today?"

    if re.search(r"\bgood afternoon\b", text):
        return "Good Afternoon 🌤 How can I assist you with admissions today?"

    if re.search(r"\bgood evening\b", text):
        return "Good Evening 🌙 How can I assist you with admissions today?"

    if re.search(r"\b(hi+|hello+|hey+)\b", text):
        return "Hello 👋 How can I assist you with admissions today?"

    return None

ADMISSION_KEYWORDS = [
    "admission", "eligibility", "fees", "fee", "documents",
    "process", "criteria", "cutoff", "program", "course",
    "intake", "application", "scholarship"
]

def admission_assistant(user_query):

    query_lower = user_query.lower().strip()

    # ---- Greeting Detection ----
    greeting_response = detect_greeting(user_query)
    if greeting_response:
        return greeting_response

    # ---- Conversation Memory (Last 5 Messages) ----
    conversation_memory = ""
    if "messages" in st.session_state:
        last_messages = st.session_state.messages[-5:]
        conversation_memory = "\n".join(
            [f'{msg["role"]}: {msg["content"]}' for msg in last_messages]
        )

    # ---- Retrieve Relevant Chunks FIRST (Smart Domain Check) ----
    retrieved_nodes = retriever.retrieve(user_query)

    # ---- Follow-up Detection ----
    followup = is_followup(query_lower)

    # ---- If Nothing Retrieved AND Not Follow-up → Block ----
    if not retrieved_nodes and not followup:
        return "Please ask questions related to SVERI college only."

    # ---- If Follow-up but Nothing Retrieved ----
    # Use conversation memory to guide answer
    if not retrieved_nodes and followup:
        refined_context = conversation_memory
        
    else:
    # ---- Re-ranking Top 3 ----
    top_3_nodes = sorted(
        retrieved_nodes,
        key=lambda x: x.score if x.score else 0,
        reverse=True
    )[:3]

    # ---- Metadata Validation (SVERI Only) ----
    if not all(node.node.metadata.get("college") == "SVERI" for node in top_3_nodes):
        return "I provide information only about SVERI college."

    refined_context = "\n\n".join(
        [node.node.text for node in top_3_nodes]
    )

    # ---- Strict Prompt ----
    prompt = f"""
    You are the official AI Assistant of SVERI College.

    STRICT RULES:
    - Answer ONLY using the provided context.
    - Use previous conversation if needed.
    - If answer is not present in context, say:
      "The information is not available in the provided documents."
    - Do NOT provide general knowledge.
    - Keep answers clear and concise in bullet points.

    Previous Conversation:
    {conversation_memory}

    Context:
    {refined_context}

    Current Question:
    {user_query}
    """

    response = llm.complete(prompt)
    return response.text

FOLLOWUP_KEYWORDS = [
    "tell me more", "more about", "explain more",
    "details", "elaborate", "continue", "what about it",
    "its details", "more info"
]

def is_followup(query):
    return any(keyword in query for keyword in FOLLOWUP_KEYWORDS)

# Streamlit UI
st.set_page_config(page_title="Admission Assistant")
st.title("🎓 SVERI Q&A Assistant")

# SIDEBAR
with st.sidebar:
    st.header("🎓 Admission Help Desk")

    st.write("""
    This assistant helps you with questions about college admissions.

    You can ask about:
             
    • Eligibility criteria  
    • Fee structure  
    • Required documents  
    • Admission process  
    • Available programs  

    Simply type your question in the chat box.
    """)

    st.markdown("---")

    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("AI-powered Admission Q&A Assistant")
        
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Generate assistant response
    with st.chat_message("assistant"):
        response = admission_assistant(prompt)
        st.markdown(response)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )








