import json
import streamlit as st
from openai import OpenAI
import pandas as pd
from utils.constants import *
from utils.streamlit_utils import *
import sys
import os

# Now you can import the aggregate_tables module
from aggregate_tables import aggregate_tables

def display_welcome_message():
    st.markdown("""
        ## 🤖 Welcome to AutoPop ChatBot!
        
        I am your assistant, here to help you with various tasks:
        
        - **Fetch and aggregate tables**: Just ask me to get a table of something!
        - **General Questions**: Ask me anything else, and I'll try my best to assist.
        
        **How to use:**
        - Type your request in the input box below.
        - You can ask for a table by starting with "Get me a table of...".
        - Or just chat with me to get started!
        
        ### What would you like to do today?
        """)
    if st.button("Get a table"):
        prompt = "Get me a table of "
        return prompt  # Pre-fill with a table request
    return None

def process_dataframe_query(prompt, df):
    """Use OpenAI to translate the prompt into a DataFrame operation."""
    # Construct a prompt to send to OpenAI to parse the user prompt into a Pandas command
    openai_prompt = f"Given this DataFrame: {df}, Answer this query: '{prompt}'."
    # Call OpenAI to interpret the user's query
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": openai_prompt},
        ],
    )
    response = response.choices[0].message.content.strip()
    return response

if __name__ == "__main__":

    # Title Container
    with st.container():
        st.title("AutoPop ChatBot")

    # Load the API keys
    with open(KEYFILE_LOC) as f:
        dic_keys = json.load(f)
        client = OpenAI(api_key=dic_keys["openAI_token"])

    # Initialize session state variables
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"

    if "delete_flag" not in st.session_state:
        st.session_state.delete_flag = False

    # Initialize messages for the current chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        st.session_state.messages = st.session_state.chats.get(st.session_state.current_chat, [])

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

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display welcome message if no messages
    if not st.session_state.messages:
        default_prompt = display_welcome_message()
    else:
        default_prompt = None

    # Get the user's prompt
    prompt = st.chat_input("Get me a table of...") if not default_prompt else default_prompt
    print(prompt)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if the prompt is requesting a table
        if prompt.strip().lower().startswith("get me a table of"):
            # Call the aggregate-tables script
            table_output = aggregate_tables(prompt)  # Assuming this returns a DataFrame

            if table_output is not None:
                # Store the DataFrame in session state
                st.session_state.last_dataframe = table_output

                # Display the DataFrame to the user
                with st.chat_message("assistant"):
                    st.dataframe(table_output)

                st.session_state.messages.append({"role": "assistant", "content": "Displayed the requested table."})
            else:
                with st.chat_message("assistant"):
                    st.markdown("No table was generated.")

        else:
            # Check if a DataFrame exists in memory for follow-up queries
            df = st.session_state.get("last_dataframe", None)
            if df is not None:
                # Attempt to process the DataFrame query
                response = process_dataframe_query(prompt, df)
                st.dataframe(df)
            else:
                response = "No DataFrame available for reference."

            if not response:
                # General query handling with OpenAI's API
                stream = client.chat_completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)

            with st.chat_message("assistant"):
                if isinstance(response, pd.DataFrame):
                    st.dataframe(response)
                else:
                    st.markdown(response)

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
