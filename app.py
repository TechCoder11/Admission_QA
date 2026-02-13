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

for doc in documents:
    doc.metadata = {"college": "SVERI"}

index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=8)

# -------------------- UTIL FUNCTIONS --------------------

def detect_greeting(text):
    text = text.lower().strip()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    if "good morning" in text:
        return "Good Morning ☀ How can I assist you with SVERI admissions today?"
    if "good afternoon" in text:
        return "Good Afternoon 🌤 How can I assist you with SVERI admissions today?"
    if "good evening" in text:
        return "Good Evening 🌙 How can I assist you with SVERI admissions today?"
    if re.search(r"\b(hi+|hello+|hey+)\b", text):
        return "Hello 👋 How can I assist you with SVERI admissions today?"

    return None


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
            if "thank" in phrase:
                return "You're welcome 😊 I'm glad I could help. If you have more questions about SVERI, feel free to ask!"
            elif "bye" in phrase or "goodbye" in phrase or "see you" in phrase:
                return "Goodbye 👋 Have a great day! If you need any information about SVERI admissions, I'm here to help."
            elif "welcome" in phrase:
                return "😊 Always happy to assist you with SVERI information!"
            else:
                return "I'm glad I could help 😊 Let me know if you need anything else about SVERI."

    return None


FOLLOWUP_KEYWORDS = [
    "tell me more", "more about", "explain more",
    "details", "elaborate", "continue",
    "its details", "more info"
]


def is_followup(query):
    return any(keyword in query for keyword in FOLLOWUP_KEYWORDS)


def is_identity_query(query):
    query = query.lower().strip()

    identity_phrases = [
        "which college is this",
        "what is the college name",
        "what college is this",
        "which university is this",
        "what institute is this",
        "where am i",
        "where is sveri located",
        "location of sveri"
    ]

    return any(phrase in query for phrase in identity_phrases)


# -------------------- MAIN ASSISTANT FUNCTION --------------------

def admission_assistant(user_query):

    query_lower = user_query.lower().strip()

    # ---- Block Other Institutions (Smart Blocking) ----
    if "sveri" not in query_lower:
        other_college_pattern = r"""\b(
        iit|nit|vnit|mit|mitwpu|rit|coep|vjti|pict|vit|spit|dj\s*sanghvi|
        wce|walchand|ict|iiit|iiitm|gcoea|gcoe|pccoe|cummins|
        bharati\s*vidyapeeth|sinhgad|modern\s*college|
        dy\s*patil|ramrao\s*adik|nmims|symbiosis|
        sandip|kjsieit|thadomal|atharva|
        gh\s*raisoni|prmitr|pvg|aissms|
        harvard|stanford|oxford|cambridge
        )\b"""
        if re.search(other_college_pattern, query_lower):
            return "I provide information only about SVERI college."

    # ---- Greeting ----
    greeting = detect_greeting(user_query)
    if greeting:
        return greeting

    # ---- Closing ----
    closing = detect_closing(user_query)
    if closing:
        return closing

    # ---- Identity Queries ----
    if is_identity_query(query_lower):
        return """🎓 This assistant provides information for:

• College Name: SVERI College of Engineering  
• Location: Pandharpur, Maharashtra  
• Institute: Shri Vithal Education & Research Institute (SVERI)

You can ask about admissions, eligibility, fees, and programs.
"""

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

    if not retrieved_nodes and not followup:
        return "Please ask questions related to SVERI admissions."

    if not retrieved_nodes and followup:
        refined_context = conversation_memory
    else:
        top_3_nodes = sorted(
            retrieved_nodes,
            key=lambda x: x.score if x.score else 0,
            reverse=True
        )[:3]

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
    formatted_response = response.text.replace("•", "\n- ")
    return formatted_response


# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="Admission Assistant")
st.title("🎓 SVERI Q&A Assistant")

with st.sidebar:
    st.header("🎓 Admission Help Desk")

    st.write("""
This assistant helps you with questions about SVERI admissions.

You can ask about:
• Eligibility criteria  
• Fee structure  
• Required documents  
• Admission process  
• Available programs  
""")

    st.markdown("---")

    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.caption("AI-powered Admission Q&A Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("assistant"):
        response = admission_assistant(prompt)
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

