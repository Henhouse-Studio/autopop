import streamlit as st
from utils.constants import *
from utils.streamlit_utils import (
    load_api_keys,
    initialize_session_state,
    render_sidebar,
    display_chat_messages,
    display_welcome_message,
    process_prompt,
    auto_save_chat,
)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="AutoPop Chat",
        layout="wide",
        page_icon=FAVICON_LOC,
        initial_sidebar_state="auto",
    )

    # Load the OpenAI API
    client = load_api_keys()

    # Setup the page and its functions
    initialize_session_state()
    render_sidebar()
    display_chat_messages()

    # Handle the prompting
    default_prompt = (
        display_welcome_message() if not st.session_state.messages else None
    )
    prompt = (
        st.chat_input("Get me a table of...") if not default_prompt else default_prompt
    )

    if prompt:
        process_prompt(prompt, client)
        auto_save_chat(client)

    if st.session_state.delete_flag:

        st.session_state.delete_flag = False
        st.rerun()


if __name__ == "__main__":
    main()
