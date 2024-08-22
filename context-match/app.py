import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from utils.constants import *
from utils.streamlit_utils import *
from aggregate_tables import aggregate_tables


if __name__ == "__main__":

    # Page Configuration
    st.set_page_config(
        page_title="AutoPop Chat",
        layout="wide",
        page_icon=FAVICON_LOC,
        initial_sidebar_state="auto",
    )

    # Title Container
    # with st.container():
    #     st.title("AutoPop ChatBot")

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
        st.session_state.messages = st.session_state.chats.get(
            st.session_state.current_chat, []
        )

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
    st.subheader(f"{st.session_state.current_chat}", divider="gray")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Check if the message contains a DataFrame
            if "dataframe" in message:
                df = pd.DataFrame(message["dataframe"])
                st.dataframe(df)

    # Display welcome message if no messages
    if not st.session_state.messages:
        default_prompt = display_welcome_message()
    else:
        default_prompt = None

    # Get the user's prompt
    prompt = (
        st.chat_input("Get me a table of...") if not default_prompt else default_prompt
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if the prompt is requesting a table
        if prompt.strip().lower().startswith("get me a table of"):
            # Call the aggregate-tables script
            df = aggregate_tables(prompt)  # Assuming this returns a DataFrame

            if df is not None:
                # Store the DataFrame in session state
                st.session_state.last_dataframe = df

                # Store the DataFrame in the chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Here's the table you requested.",
                    "dataframe": df.to_dict()  # Save the DataFrame as a dictionary
                })

                # Display the DataFrame to the user
                with st.chat_message("assistant"):
                    st.dataframe(df)

            else:
                with st.chat_message("assistant"):
                    st.markdown("No table was generated.")

        else:
            # Check if a DataFrame exists in memory for follow-up queries
            df = st.session_state.get("last_dataframe", None)
            if df is not None:
                # Attempt to process the DataFrame query
                response = process_dataframe_query(prompt, df)

                if isinstance(response, pd.DataFrame):
                    # If the response is a DataFrame, update the chat history with it
                    st.session_state.last_dataframe = response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Here is the updated table based on your query:",
                        "dataframe": response.to_dict()  # Save the updated DataFrame as a dictionary
                    })

                    with st.chat_message("assistant"):
                        st.dataframe(response)

                else:
                    # Otherwise, treat the response as a text response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                    with st.chat_message("assistant"):
                        st.markdown(response)
            else:
                # If no DataFrame exists, handle the query as a general text-based query
                response = None
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
