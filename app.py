import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
documents = SimpleDirectoryReader(input_dir="data").load_data()

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=8)

# Query function
def admission_assistant(user_query):
    retrieved_nodes = retriever.retrieve(user_query)

    if not retrieved_nodes:
        return "No relevant information found."

    top_3_nodes = sorted(
        retrieved_nodes,
        key=lambda x: x.score if x.score else 0,
        reverse=True
    )[:3]

    refined_context = "\n\n".join(
        [node.node.text for node in top_3_nodes]
    )

    prompt = f"""
    You are an official Admission Assistant.
    Use only this context.

    Context:
    {refined_context}

    Question:
    {user_query}

    Answer in bullet points.
    """

    response = llm.complete(prompt)
    return response.text

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

#what is the eligibility criteria?

#What is the fee structure 

