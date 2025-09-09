import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please define it in your .env file.")

# External client and model
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Agent
agent = Agent(
    name="Assistant",
    instructions="I am not just a writer, I am the architect of stories that turn ideas into unforgettable scripts.",
    model=config.model,
    tools=[]
)

# Streamlit page setup
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("🤖 Welcome to the ChatBot!")
st.caption("How can I help you today?")

# Sidebar clear button
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to stream agent reply
async def get_agent_reply(user_message):
    history = st.session_state.chat_history.copy()
    history.append({"role": "user", "content": user_message})

    result = Runner.run_streamed(agent, history, run_config=config)

    reply_text = ""
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                token = event.data.delta
                reply_text += token
                msg_placeholder.markdown(reply_text + "▌")
        msg_placeholder.markdown(reply_text)

    history.append({"role": "assistant", "content": reply_text})
    st.session_state.chat_history = history

# Render conversation from history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message here..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    asyncio.run(get_agent_reply(user_input))
