import json
import streamlit as st
from openai import OpenAI


if __name__ == "__main__":

    # Set the screen title
    st.title("AutoPop ChatBot")

    # Load API keys
    with open("llamaindex/keys.json") as f:
        dic_keys = json.load(f)
        client = OpenAI(api_key=dic_keys["openAI_token"])

    # Initialize session state variables
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = "New Chat"

    # Sidebar for conversation management
    with st.sidebar:
        st.title("Conversations")

        # Button to start a new conversation
        if st.button("New Chat"):
            st.session_state.current_conversation = "New Chat"
            st.session_state.messages = []

        # Display saved conversations
        for title in st.session_state.conversations.keys():
            if st.button(title):
                st.session_state.current_conversation = title
                st.session_state.messages = st.session_state.conversations[title]

        # Input for saving the current conversation
        new_title = st.text_input("Save current chat as:")
        if new_title and st.button("Save"):
            st.session_state.conversations[new_title] = st.session_state.messages
            st.session_state.current_conversation = new_title

    # Main chat interface
    st.subheader(f"Current Conversation: {st.session_state.current_conversation}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Update the current conversation in the saved conversations
        if st.session_state.current_conversation != "New Chat":
            st.session_state.conversations[st.session_state.current_conversation] = (
                st.session_state.messages
            )
