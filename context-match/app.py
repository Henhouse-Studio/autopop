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

def show_librarian():
    """Display the current chat functionality under 'Librarian'."""
    # Load the OpenAI API
    client = load_api_keys()
    
    # Avoid duplicating the sidebar, only call render_sidebar once in the main function
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

def show_settings():
    """Display the settings page to configure the model and threshold."""
    st.title("Settings")

    # Initialize a session state variable to track changes and confirmation
    if "settings_changed" not in st.session_state:
        st.session_state.settings_changed = False
    if "confirm_save" not in st.session_state:
        st.session_state.confirm_save = False

    # Model selection
    st.subheader("Model Selection")
    selected_model = st.selectbox(
        "Choose a model:",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
        index=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"].index(st.session_state["openai_model"]),
        key="model_selectbox"
    )

    # Model Encoder
    st.subheader("Model Encoder")
    selected_model_encoder = st.selectbox(
        "Choose a model encoder:",
        ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2"],
        index=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2"].index(st.session_state["model_encoder"]),
        key="model_encoder_selectbox"
    )

    # Matching Threshold
    st.subheader("Matching Threshold")
    selected_matching_threshold = st.slider(
        "Set the matching threshold:",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["matching_threshold"]),
        step=0.01,
        key="matching_threshold_slider"
    )

    # Tolerance
    st.subheader("Tolerance")
    selected_tolerance = st.slider(
        "Set the tolerance:",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["tolerance"]),
        step=0.01,
        key="tolerance_slider"
    )

    # Temperature
    st.subheader("Temperature")
    selected_temperature = st.slider(
        "Set the temperature:",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["temperature"]),
        step=0.01,
        key="temperature_slider"
    )

    # Check if settings have been changed
    if (selected_model != st.session_state["openai_model"] or
        selected_matching_threshold != st.session_state["matching_threshold"] or
        selected_tolerance != st.session_state["tolerance"] or
        selected_model_encoder != st.session_state["model_encoder"] or
        selected_temperature != st.session_state["temperature"]):
        st.session_state.settings_changed = True

    # Centering the Save button with custom styling
    if st.session_state.settings_changed:
        st.write("")
        save_button = st.button("💾 Save Settings", key="save_settings_button")
        if save_button:
            st.session_state["openai_model"] = selected_model
            st.session_state["matching_threshold"] = selected_matching_threshold
            st.session_state["tolerance"] = selected_tolerance
            st.session_state["model_encoder"] = selected_model_encoder
            st.session_state["temperature"] = selected_temperature
            st.session_state.settings_changed = False
            st.success("Settings updated successfully!")

        # Custom CSS for the specific button
        st.markdown(
            """
            <style>
            #save_settings_button button {
                background-color: #4CAF50;
                color: white;
                padding: 0.5em 1em;
                font-size: 18px;
                border-radius: 8px;
                border: 2px solid #4CAF50;
                transition: 0.3s;
            }
            #save_settings_button button:hover {
                background-color: white;
                color: #4CAF50;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="AutoPop Chat",
        layout="wide",
        page_icon=FAVICON_LOC,
        initial_sidebar_state="auto",
    )

    # Initialize chats in session state
    initialize_session_state()

    # Sidebar setup
    render_sidebar()

    # Only display navigation buttons if no page is selected
    if "page" not in st.session_state:
        st.session_state.page = None

    if st.session_state.page is None:
        st.markdown(
            """
            <style>
            .block-container {
                display: block;
                justify-content: center;
                align-items: center;
                align-content: center;
                height: 100vh;
            }

            .centered-container {
                display: flex;
                justify-content: center;
                align-items: center;
                # height: 100vh;  /* Full viewport height */
                flex-direction: column;
                padding: 0;
                margin: 0;
                width: 100% !important;
            }
            h1 {
                font-size: 2.5em;
                margin: 0;
                padding: 0;
                text-align: center;
            }
            h4 {
                font-size: 1.3m;
                margin: 0;
                padding-top: 5px;
                padding-bottom: 0px;
                text-align: center;
                color: darkgray;
                font-weight: 500;
            }
            .button-container {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                gap: 0.5em;
                margin-top: 2em;
                width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
        st.markdown("<h1>Welcome to AutoPop Chat</h1>", unsafe_allow_html=True)
        st.markdown("<h4>By Henhouse Studio </h4>", unsafe_allow_html=True)

        st.markdown("<div class='button-container'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1,1,1,1])

        with col1:
            if st.button("⚡️ Start", key="start_button", use_container_width=True):
                st.session_state.page = "Start"
                st.rerun()

        with col2:
            if st.button("📚 Librarian", key="librarian_button", use_container_width=True):
                st.session_state.page = "Librarian"
                st.rerun()

        with col3:
            if st.button("💾 Databases", key="databases_button", use_container_width=True):
                st.session_state.page = "Databases"
                st.rerun()

        with col4:
            if st.button("⚙️ Settings", key="settings_button", use_container_width=True):
                st.session_state.page = "Settings"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Display content based on the selected page
    if st.session_state.page == "Start":
        st.title("Welcome to AutoPop Chat")

    elif st.session_state.page == "Librarian":
        show_librarian()

    elif st.session_state.page == "Databases":
        st.title("Databases")
        st.write("Database-related functionality will go here.")

    elif st.session_state.page == "Settings":
        show_settings()

if __name__ == "__main__":
    main()