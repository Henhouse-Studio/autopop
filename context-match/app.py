import pandas as pd
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

import os
import pandas as pd
import streamlit as st

def show_databases(data_dir="path_to_data_dir"):
    """
    Display the tables stored as CSV files in the specified directory, including tags.

    :param data_dir: The directory where the CSV files are stored.
    """
    st.title("Databases")

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        st.error(f"The directory {data_dir} does not exist.")
        return

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

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
            modified_date = pd.to_datetime(modified_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.error(f"Error loading {csv_file}: {e}")
            modified_date = "N/A"

        # Get tags based on the file name, or "None" if not found
        tags = tags_mapping.get(file_name, ["None"])

        # Add the row to the data list
        data.append({
            "Index": i + 1,
            "Name": csv_file,
            "Tags": ", ".join(tags),  # Convert list of tags to a comma-separated string
            "Modified": modified_date,
        })

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
                    "Modified": st.column_config.TextColumn("Last Modified")
                },
                hide_index=True,
                use_container_width=True  # This makes the table occupy the full width
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

# You can now call the function with your specific directory:
# show_databases("your_directory_path_here")




# Example usage:
# show_databases(data_dir="your_data_directory_path")


# Example usage:
# show_databases(data_dir="your_data_directory_path")

 # Method 1
    # # Create a table with each row representing a CSV file
    # for i, csv_file in enumerate(csv_files):
    #     file_path = os.path.join(data_dir, csv_file)
    #     try:
    #         # Load the CSV file into a DataFrame
    #         df = pd.read_csv(file_path)
            
    #         # Use an expander to display the content of the table
    #         with st.expander(f"Table {i+1}: {csv_file}"):
    #             st.dataframe(df)
                
    #     except Exception as e:
    #         st.error(f"Error loading {csv_file}: {e}")

    # Method 2
    # # Dropdown to select a CSV file
    # selected_file = st.selectbox("Select a table to view:", csv_files)

    # if selected_file:
    #     file_path = os.path.join(data_dir, selected_file)
    #     try:
    #         # Load the selected CSV file into a DataFrame
    #         df = pd.read_csv(file_path)
    #         # Display the DataFrame
    #         st.subheader(f"Content of {selected_file}")
    #         st.dataframe(df)
    #     except Exception as e:
    #         st.error(f"Error loading {selected_file}: {e}")

    # WORKING
    # if 'visible_tables' not in st.session_state:
    #     st.session_state.visible_tables = {csv_file: False for csv_file in csv_files}

    # # Function to toggle visibility
    # def toggle_visibility(csv_file):
    #     st.session_state.visible_tables[csv_file] = not st.session_state.visible_tables[csv_file]

    # # Iterate through the list of CSV files and create a row for each file
    # for i, csv_file in enumerate(csv_files):
    #     col1, col2, col3 = st.columns([1, 6, 2])  # Adjust column widths
    #     col1.write(i + 1)  # Index
    #     col2.write(csv_file)  # File name

    #     # Create a button that toggles visibility
    #     button_label = "Hide Table" if st.session_state.visible_tables[csv_file] else "View Table"
    #     if col3.button(button_label, key=f"toggle_{csv_file}", on_click=toggle_visibility, args=(csv_file,)):
    #         pass  # The on_click function will handle the state change

    #     # If the table is set to be visible, display it
    #     if st.session_state.visible_tables[csv_file]:
    #         file_path = os.path.join(data_dir, csv_file)
    #         try:
    #             df = pd.read_csv(file_path)
    #             # st.write(f"Content of {csv_file}")
    #             st.dataframe(df)
    #         except Exception as e:
    #             st.error(f"Error loading {csv_file}: {e}")


#     # Initialize session state for each CSV file to track visibility
#     if 'visible_tables' not in st.session_state:
#         st.session_state.visible_tables = {csv_file: False for csv_file in csv_files}

#     # Function to toggle visibility
#     def toggle_visibility(csv_file):
#         st.session_state.visible_tables[csv_file] = not st.session_state.visible_tables[csv_file]

#    # CSS for the layout
#     st.markdown("""
#     <style>
#     .index-text {
#         text-align: left;
#         padding-right: 1em;
#         color: darkgray;
#     }
#     .file-row {
#         display: flex;
#         align-items: center;
#         margin-bottom: 0em;
#     }
#     .file-name {
#         flex-grow: 1;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     # Iterate through the list of CSV files and create a row for each file
#     for i, csv_file in enumerate(csv_files):
#         col1, col2 = st.columns([6, 2])
        
#         with col1:
#             st.markdown(f"""
#             <div class="file-row">
#                 <span class="index-text">{i + 1}</span>
#                 <span class="file-name">{csv_file}</span>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             button_label = "Hide Table" if st.session_state.visible_tables[csv_file] else "View Table"
#             if st.button(button_label, key=f"toggle_{csv_file}", on_click=toggle_visibility, args=(csv_file,)):
#                 pass  # The on_click function will handle the state change

#         # If the table is set to be visible, display it
#         if st.session_state.visible_tables[csv_file]:
#             file_path = os.path.join(data_dir, csv_file)
#             try:
#                 df = pd.read_csv(file_path)
#                 st.dataframe(df)
#             except Exception as e:
#                 st.error(f"Error loading {csv_file}: {e}")
    



def show_databases_(data_dir="path_to_data_dir"):
    """
    Display the tables stored as CSV files in the specified directory.

    :param data_dir: The directory where the CSV files are stored.
    """
    st.title("Databases")

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        st.error(f"The directory {data_dir} does not exist.")
        return

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        st.warning("No CSV files found in the directory.")
        return

    # Initialize session state for each CSV file to track visibility
    if 'visible_tables' not in st.session_state:
        st.session_state.visible_tables = {csv_file: False for csv_file in csv_files}

    # Function to toggle visibility
    def toggle_visibility(csv_file):
        st.session_state.visible_tables[csv_file] = not st.session_state.visible_tables[csv_file]

    # CSS for the layout
    st.markdown("""
    <style>

    .index-text {
        text-align: left;
        padding-right: 1em;
        color: darkgray;
    }
    .file-row {
        display: flex;
        align-items: center;
        margin-bottom: 0em;
    }
    .file-name {
        flex-grow: 1;
    }
    .header-row {
        display: flex;
        align-items: center;
        font-weight: bold;
    }
    .header {
        border-bottom: 1px solid #ccc;
        padding-bottom: 0.5em;
        margin-bottom: 0.5em;
    }
    </style>
    """, unsafe_allow_html=True)

    columns_spacing = [1, 5, 2, 2]
    row_cols = st.columns(columns_spacing)

    st.markdown(f"<div class='header'>", unsafe_allow_html=True)   
    with row_cols[0]:
        st.markdown(f"<div class='header-row'>Index</div>", unsafe_allow_html=True)
    with row_cols[1]:
        st.markdown(f"<div class='header-row'>Name</div>", unsafe_allow_html=True)
    with row_cols[2]:
        st.markdown(f"<div class='header-row'>Modified</div>", unsafe_allow_html=True)
    with row_cols[3]:
        st.markdown(f"<div class='header-row'>Action</div>", unsafe_allow_html=True)
    st.markdown(f"</div>", unsafe_allow_html=True)

    # Iterate through the list of CSV files and create a row for each file
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(data_dir, csv_file)
        
        # Get the first row to extract column names and modified time
        try:
            df = pd.read_csv(file_path, nrows=1)
            modified_time = os.path.getmtime(file_path)  # Get last modified time
            modified_date = pd.to_datetime(modified_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.error(f"Error loading {csv_file}: {e}")
            modified_date = "N/A"

        row_cols = st.columns(columns_spacing)
        
        with row_cols[0]:
            st.markdown(f"<div class='index-text'>{i + 1}</div>", unsafe_allow_html=True)
        
        with row_cols[1]:
            st.markdown(f"<div class='file-name'>{csv_file}</div>", unsafe_allow_html=True)
        
        with row_cols[2]:
            st.markdown(f"<div class='file-name'>{modified_date}</div>", unsafe_allow_html=True)
        
        with row_cols[3]:
            button_label = "Hide Table" if st.session_state.visible_tables[csv_file] else "View Table"
            if st.button(button_label, key=f"toggle_{csv_file}", on_click=toggle_visibility, args=(csv_file,)):
                pass  # The on_click function will handle the state change

        # If the table is set to be visible, display it
        if st.session_state.visible_tables[csv_file]:
            try:
                df = pd.read_csv(file_path)
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error loading {csv_file}: {e}")

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
        show_databases(data_dir='databases/table_of_tables/')

    elif st.session_state.page == "Settings":
        show_settings()

if __name__ == "__main__":
    main()