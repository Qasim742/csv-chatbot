import os
import dotenv
import streamlit as st
import pandas as pd

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent
)

dotenv.load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="CSV Chatbot",
    page_icon="🤖"
)

st.title("CSV Chatbot")


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=api_key,
    temperature=0
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("CSV Upload")

    uploaded_file = st.file_uploader(
        "Upload your CSV",
        type=["csv"]
    )


# -----------------------------
# Load CSV
# -----------------------------

df = None
agent = None

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("CSV Data")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.write(
        f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="tool-calling"
    )


# -----------------------------
# Display Previous Messages
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

prompt = st.chat_input(
    "Ask me anything..."
)


# -----------------------------
# Process Message
# -----------------------------

if prompt:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)


    # Generate response
    with st.chat_message("assistant"):

        if agent is not None:

            # CSV mode
            response = agent.invoke(prompt)

            answer = response["output"]

        else:

            # Normal chatbot mode
            response = llm.invoke(prompt)

            answer = response.content


        st.markdown(answer)


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )