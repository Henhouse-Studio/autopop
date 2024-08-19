import json
import streamlit as st
from openai import OpenAI
from utils.streamlit_utils import *


if __name__ == "__main__":

    # #  Load the CSS
    # load_css(CSS_LOC)

    # Workaround for the body, sidebar, and header/footer colors
    # (requires below formatting)
    # st.markdown(
    #     """
    #     <style>
    #     [data-testid="stSidebarContent"] {
    #         color: black;
    #         background-color: #FF9E50;
    #     }
    #     [data-testid="stSidebarContent"] {
    #         color: black;
    #         background-color: #FF9E50;
    #     }
    #     [data-testid="stAppViewContainer"] {
    #         color: brown;
    #         background-color: white;
    #     }
    #     [data-testid="stHeader"] {
    #         color: black;
    #         background-color: rgba(0, 0, 0, 0);
    #     }
    #     [data-testid="stBottom"] {
    #         color: white !important;
    #         background-color: white !important;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # Title Container
    with st.container():
        st.title("AutoPop ChatBot")

    # Load the API keys
    with open("llamaindex/keys.json") as f:
        dic_keys = json.load(f)
        client = OpenAI(api_key=dic_keys["openAI_token"])

    # Initialize session state variables
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"

    if "delete_flag" not in st.session_state:
        st.session_state.delete_flag = False

    # Sidebar for chat management
    with st.sidebar:

        st.subheader("Conversations")
        # Button to start a new chat
        if st.button("New Chat"):
            st.session_state.current_chat = "New Chat"
            st.session_state.messages = []

        # Display saved chats with delete buttons
        for title in list(st.session_state.chats.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(title, key=f"open_{title}"):
                    st.session_state.current_chat = title
                    st.session_state.messages = st.session_state.chats[title]
            with col2:
                if st.button("🗑️", key=f"delete_{title}"):
                    delete_chat(title)
                    st.session_state.delete_flag = True
                    st.rerun()

    # Main chat interface
    st.subheader(f"{st.session_state.current_chat}")
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

        # Auto-save and generate title for new chats
        if (
            st.session_state.current_chat == "New Chat"
            and len(st.session_state.messages) == 2
        ):
            new_title = generate_title(client, st.session_state.messages)
            st.session_state.current_chat = new_title
            st.session_state.chats[new_title] = st.session_state.messages
            save_chat(new_title, st.session_state.messages)
            st.rerun()

        # Update the current chat in the saved chats
        if st.session_state.current_chat != "New Chat":
            st.session_state.chats[st.session_state.current_chat] = (
                st.session_state.messages
            )
            save_chat(st.session_state.current_chat, st.session_state.messages)

    # Handle deletion rerun
    if st.session_state.delete_flag:
        st.session_state.delete_flag = False
        st.rerun()
