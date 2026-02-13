import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import re
import os

groq_api_key = os.getenv("groq_api_key")

if not groq_api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()

# -------------------- MODELS --------------------

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

# -------------------- LOAD DOCUMENTS --------------------

documents = SimpleDirectoryReader(input_dir="Data").load_data()

# Add metadata
for doc in documents:
    doc.metadata = {"college": "SVERI"}

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=8)

# -------------------- UTIL FUNCTIONS --------------------

def detect_greeting(text):
    text = text.lower().strip()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    if "good morning" in text:
        return "Good Morning ☀ How can I assist you with admissions today?"
    if "good afternoon" in text:
        return "Good Afternoon 🌤 How can I assist you with admissions today?"
    if "good evening" in text:
        return "Good Evening 🌙 How can I assist you with admissions today?"
    if re.search(r"\b(hi+|hello+|hey+)\b", text):
        return "Hello 👋 How can I assist you with admissions today?"

    return None


FOLLOWUP_KEYWORDS = [
    "tell me more", "more about", "explain more",
    "details", "elaborate", "continue", "what about it",
    "its details", "more info"
]

def is_followup(query):
    return any(keyword in query for keyword in FOLLOWUP_KEYWORDS)


def detect_closing(text):
    text = text.lower().strip()

    closing_patterns = [
        "bye", "goodbye", "see you",
        "thank you", "thanks", "thx",
        "ok thanks", "okay thanks",
        "that's helpful", "got it",
        "welcome"
    ]

    for phrase in closing_patterns:
        if phrase in text:
            if "thank" in phrase or "thanks" in phrase:
                return "You're welcome 😊 I'm glad I could help. If you have more questions about SVERI, feel free to ask!"
            elif "bye" in phrase or "goodbye" in phrase or "see you" in phrase:
                return "Goodbye 👋 Have a great day! If you need any information about SVERI admissions, I'm here to help."
            elif "welcome" in phrase:
                return "😊 Always happy to assist you with SVERI information!"
            else:
                return "I'm glad I could help 😊 Let me know if you need anything else about SVERI."

    return None

def is_identity_query(query):
    query = query.lower().strip()

    identity_phrases = [
        "which college is this",
        "what is the college name",
        "what college is this",
        "which university is this",
        "what institute is this",
        "where am i"
    ]

    return any(phrase in query for phrase in identity_phrases)
# -------------------- MAIN ASSISTANT FUNCTION --------------------

def admission_assistant(user_query):
    
# Explicitly block other institution names
    blocked_institutions = [
    "mit", "iit", "nit", "harvard", "vnit",
    "coep", "stanford", "oxford", "cambridge",
    "rit","wit","wce","ICT","VJTI", "PICT","SPIT", "VIT" ,
    "D.J. Sanghvi College of Engineering","MIT-WPU","Cummins College of Engineering for Women"
    ]

query_lower = user_query.lower()

if any(name in query_lower for name in blocked_institutions):
    return "I provide information only about SVERI college."
    query_lower = user_query.lower().strip()

      # ---- Greeting ----
    greeting = detect_greeting(user_query)
    if greeting:
        return greeting

    # ---- Closing Detection ----
    closing = detect_closing(user_query)
    if closing:
        return closing
        
    # ---- Conversation Memory ----
    conversation_memory = ""
    if "messages" in st.session_state:
        last_messages = st.session_state.messages[-5:]
        conversation_memory = "\n".join(
            [f'{msg["role"]}: {msg["content"]}' for msg in last_messages]
        )

    # ---- Retrieve ----
    retrieved_nodes = retriever.retrieve(user_query)

    followup = is_followup(query_lower)

    # ---- If nothing retrieved ----
    if not retrieved_nodes and not followup:
        return "Please ask questions related to SVERI college only."

    # ---- Handle followup without retrieval ----
    if not retrieved_nodes and followup:
        refined_context = conversation_memory
    else:
        top_3_nodes = sorted(
            retrieved_nodes,
            key=lambda x: x.score if x.score else 0,
            reverse=True
        )[:3]

        # Metadata validation
        if not all(node.node.metadata.get("college") == "SVERI" for node in top_3_nodes):
            return "I provide information only about SVERI college."

        refined_context = "\n\n".join(
            [node.node.text for node in top_3_nodes]
        )

    # ---- Prompt ----
 prompt = f"""
You are the official AI Assistant of SVERI College.


STRICT RULES:
- Answer ONLY using the retrieved context.
- Do NOT use outside knowledge.
- Do NOT guess.
- If answer is not present in the context, say exactly:
  "The requested information is not available in official SVERI documents."


    Previous Conversation:
    {conversation_memory}

    Context:
    {refined_context}

    Current Question:
    {user_query}

    Answer in bullet points:
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













