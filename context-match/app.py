import pandas as pd
import streamlit as st
from utils.constants import *
from utils.streamlit_utils import *


def show_librarian():

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

    elif st.session_state.prompt != "":
        process_prompt(st.session_state.prompt, client)
        auto_save_chat(client)

    if st.session_state.delete_flag:
        st.session_state.delete_flag = False
        st.rerun()


def show_databases(data_dir="path_to_data_dir"):

    st.title("Databases")

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        st.error(f"The directory {data_dir} does not exist.")
        return

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        st.warning("No CSV files found in the directory.")
        return

    # Create a dataframe to hold file information
    data = []
    # Define tags based on file names (You may need to adjust these mappings)
    tags_mapping = {
        "LinkedIn Profiles": ["People", "Work"],
        "Blog Profiles": ["People", "Hobbies"],
        "Company Profiles": ["Work"],
        "Locations - Netherlands": ["Fact", "Location"],
        "Positions - Netherlands": ["Work", "Fact"],
        "Universities and Programmes - Netherlands": ["Education", "Fact"],
    }

    for i, csv_file in enumerate(csv_files):
        file_name = os.path.splitext(csv_file)[0]  # Remove the .csv extension
        file_path = os.path.join(data_dir, csv_file)

        # Get the modified time
        try:
            modified_time = os.path.getmtime(file_path)
            modified_date = pd.to_datetime(modified_time, unit="s").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception as e:
            st.error(f"Error loading {csv_file}: {e}")
            modified_date = "N/A"

        # Get tags based on the file name, or "None" if not found
        tags = tags_mapping.get(file_name, ["None"])

        # Add the row to the data list
        data.append(
            {
                "Index": i + 1,
                "Name": csv_file,
                "Tags": ", ".join(
                    tags
                ),  # Convert list of tags to a comma-separated string
                "Modified": modified_date,
            }
        )

    # Selector to choose "All Tables" or a specific table
    options = ["All Tables"] + csv_files
    selected_option = st.selectbox("Select a table to view:", options)

    if selected_option == "All Tables":
        # Create the dataframe
        file_df = pd.DataFrame(data)

        # Display the dataframe with interactive sorting, occupying the full width available
        with st.container():
            st.data_editor(
                file_df,
                column_config={
                    "Name": st.column_config.TextColumn("File Name"),
                    "Tags": st.column_config.TextColumn("Tags"),
                    "Modified": st.column_config.TextColumn("Last Modified"),
                },
                hide_index=True,
                use_container_width=True,  # This makes the table occupy the full width
            )
    else:
        # Display the selected table
        file_path = os.path.join(data_dir, selected_option)
        try:
            # Load the selected CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Display the DataFrame
            st.subheader(f"Content of {selected_option}")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading {selected_option}: {e}")


def show_settings():
    """Display the settings page to configure the model and threshold."""
    st.title("Settings")

    load_css(STYLE_SETTINGS)

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
        index=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"].index(
            st.session_state["openai_model"]
        ),
        key="model_selectbox",
    )

    # Model Encoder
    st.subheader("Model Encoder")
    selected_model_encoder = st.selectbox(
        "Choose a model encoder:",
        ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2"],
        index=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2"].index(
            st.session_state["model_encoder"]
        ),
        key="model_encoder_selectbox",
    )

    # Matching Threshold
    st.subheader("Matching Threshold")
    selected_matching_threshold = st.slider(
        "Set the matching threshold:",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["matching_threshold"]),
        step=0.01,
        key="matching_threshold_slider",
    )

    # Tolerance
    st.subheader("Tolerance")
    selected_tolerance = st.slider(
        "Set the tolerance:",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["tolerance"]),
        step=0.01,
        key="tolerance_slider",
    )

    # Temperature
    st.subheader("Temperature")
    selected_temperature = st.slider(
        "Set the temperature:",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["temperature"]),
        step=0.01,
        key="temperature_slider",
    )

    # Check if settings have been changed
    if (
        selected_model != st.session_state["openai_model"]
        or selected_matching_threshold != st.session_state["matching_threshold"]
        or selected_tolerance != st.session_state["tolerance"]
        or selected_model_encoder != st.session_state["model_encoder"]
        or selected_temperature != st.session_state["temperature"]
    ):
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


def main():
    """Main function for running the Streamlit app."""
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

    # Main Page setup
    if st.session_state.page is None:

        load_css(STYLE_MAIN)

        st.markdown(
            """<div class='centered-container'>
                    <h1>Welcome to AutoPop Chat</h1>
                    <h4>By Henhouse Studio </h4>
                    <div class='button-container'>""",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("💬 Start", key="start_button", use_container_width=True):
                st.session_state.page = "Librarian"
                st.rerun()

        with col2:
            if st.button(
                "💾 Databases", key="databases_button", use_container_width=True
            ):
                st.session_state.page = "Databases"
                st.rerun()

        with col3:
            if st.button("⚙️ Settings", key="settings_button", use_container_width=True):
                st.session_state.page = "Settings"
                st.rerun()

    # Display content based on the selected page
    if st.session_state.page == "Start":
        st.title("Welcome to AutoPop Chat")

    elif st.session_state.page == "Librarian":
        show_librarian()

    elif st.session_state.page == "Databases":
        show_databases(data_dir=DATA_DIR)

    elif st.session_state.page == "Settings":
        show_settings()


if __name__ == "__main__":
    main()
