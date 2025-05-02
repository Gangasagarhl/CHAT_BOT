# filename: app.py

import os
import streamlit as st
from typing import Dict
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.utils.token_length import get_token_length_function

# ✅ Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Smart Chatbot With Groq"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ✅ Streamlit Sidebar Controls
st.sidebar.title("Chat Settings")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# ✅ Instantiate Groq LLM
model = ChatGroq(model_name="Compound-Beta-Mini", temperature=temperature, max_tokens=max_tokens)

# ✅ Get token counter for trimming
token_counter_fn = get_token_length_function(model_name="Compound-Beta-Mini")

# ✅ Trim old messages if they exceed token limit
trimmer = trim_messages(
    max_tokens=300,
    strategy="last",
    token_counter=token_counter_fn,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# ✅ Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all the questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages")
])

# ✅ In-memory session history store
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ✅ Build the LangChain Runnable
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="messages"
)

# ✅ Streamlit UI
st.title("🤖 Smart Chatbot with Groq & LangSmith")

# Session ID Setup
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_456"
session_id = st.session_state.session_id

# Chat history initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="user_input")

# ✅ Process user input
if user_input:
    full_messages = st.session_state.chat_history + [HumanMessage(content=user_input)]

    response = chat_chain.invoke(
        input={"messages": full_messages},
        config={"configurable": {"session_id": session_id}}
    )

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response.content))

# ✅ Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Bot:** {msg.content}")

