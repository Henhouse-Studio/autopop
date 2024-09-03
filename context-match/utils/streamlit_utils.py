import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from utils.constants import *
from aggregate_tables import aggregate_tables
from utils.make_embeddings import compute_embedding
from utils.compute_similarity import compute_similarity


def display_welcome_message():

    st.markdown(
        """
        ## 🤖 Welcome to the AutoPop ChatBot!

        I am your personal data assistant, ready to help with various tasks:

        - **Fetch and aggregate tables**: Just ask me to get a table of something!
        - **General Questions**: Ask me anything else, and I'll do my best to assist!

        _**How to use:**_
        - _Type your request in the input box below._
        - _You can ask for a table by starting with "Get me a table of..."._
        - _Or just chat with me to get started!_

        #### _So, what would you like to do today?_
        """
    )

    return None


def save_chat(title, messages):

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    extension = ("." * ("." not in title)) + "json"

    file_path = os.path.join(SAVE_FOLDER, f"{title}{extension}")

    with open(file_path, "w") as f:
        json.dump(messages, f, indent=2)


def load_chats():

    chats = {}
    if os.path.exists(SAVE_FOLDER):
        for filename in os.listdir(SAVE_FOLDER):

            if filename.endswith(".json"):
                title = filename[:-5]  # Remove .json extension
                file_path = os.path.join(SAVE_FOLDER, filename)
                with open(file_path, "r") as f:
                    try:
                        # Try to load the JSON file
                        chats[title] = json.load(f)

                    except json.JSONDecodeError:
                        # If there's an error (e.g., file is empty or malformed), log it and skip the file
                        print(
                            f"Warning: Skipping file '{filename}' due to a JSONDecodeError."
                        )
                        continue

    return chats


def delete_chat(title):

    file_path = os.path.join(SAVE_FOLDER, f"{title}.json")
    if os.path.exists(file_path):
        os.remove(file_path)

    del st.session_state.chats[title]

    if st.session_state.current_chat == title:
        st.session_state.current_chat = "New Chat"
        st.session_state.messages = []


def generate_title(client, messages):

    summary_prompt = f"""
    Summarize the following chat in 8 words or less:\n\n{messages[0]['content']}\n{messages[1]['content']}.

    If 'get me a table' is in the start, focus on what type of table is being requested.

    Do not include characters that are illegal for filenames.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


def process_dataframe_query(prompt, client, df):
    """Use OpenAI to translate the prompt into a DataFrame operation."""

    # Serialize the DataFrame to ensure all columns are included, even if large
    df_serialized = df.to_json()

    # Construct a prompt to send to OpenAI to parse the user prompt into a Pandas command
    openai_prompt = f"""Given this DataFrame: {df_serialized}, Answer this query: '{prompt}'.
        
                     If the query and DataFrame are unrelated, then just ignore the DataFrame and
                     answer the query as is. Do not start with 'the answer to this query is...'."""

    # Potential TODO: Link method of merging to explanation (to explain probabilities)

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


def load_api_keys():
    """Load API keys from the specified file."""

    with open(KEYFILE_LOC) as f:
        dic_keys = json.load(f)

    return OpenAI(api_key=dic_keys["openAI_token"])


def initialize_session_state():
    """Initialize Streamlit session state variables."""

    # =========== Arguments for Navigation ===========
    if "page" not in st.session_state:
        st.session_state.page = None

    # =========== Arguments for Settings ===========
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    if "matching_threshold" not in st.session_state:
        st.session_state["matching_threshold"] = 0.14

    if "tolerance" not in st.session_state:
        st.session_state["tolerance"] = 0.1

    if "model_encoder" not in st.session_state:
        st.session_state["model_encoder"] = "all-MiniLM-L6-v2"

    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.0

    # =========== Application State Variables ===========
    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"

    if "delete_flag" not in st.session_state:
        st.session_state.delete_flag = False

    # init prompt: no messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        st.session_state.messages = st.session_state.chats.get(
            st.session_state.current_chat, []
        )

    # print("NEW")
    # print(st.session_state.messages)

    # =========== Aggregation State Variables ===========
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""

    if "process_stage" not in st.session_state:
        st.session_state.process_stage = "start"


def render_sidebar():
    """Render the chat management sidebar with a rename function."""
    with st.sidebar:

        st.header("AutoPop Chat v1.0")

        # Create a row with two columns to house the buttons side by side
        col1, col2 = st.columns([0.15, 0.9])

        with col1:
            # Add the "Go to Menu" button in the sidebar
            if st.button("🏠", key="menu_button"):
                st.session_state.page = None
                st.rerun()

        with col2:
            # Add the "New Chat" button in the sidebar
            if st.button("New Chat"):
                st.session_state.current_chat = "New Chat"
                st.session_state.messages = []
                st.session_state.page = "Librarian"
                st.rerun()

        # Display the list of previous chats
        for title in list(st.session_state.chats.keys()):
            with st.expander(f"{title}", expanded=False):
                col1, col2, col3 = st.columns([3, 2, 4])

                with col1:
                    if st.button(f"Open", key=f"open_{title}"):
                        st.session_state.current_chat = title
                        st.session_state.messages = st.session_state.chats[title]
                        st.session_state.page = (
                            "Librarian"  # Redirect to Librarian page
                        )
                        st.rerun()

                with col2:
                    if st.button("🗑️", key=f"delete_{title}"):
                        delete_chat(title)
                        st.session_state.delete_flag = True
                        st.rerun()

                with col3:
                    new_title = st.text_input(
                        f"Rename", value=title, key=f"rename_{title}"
                    )

                    if new_title != title and new_title.strip() != "":
                        # Rename chat in session state and on disk
                        st.session_state.chats[new_title] = st.session_state.chats.pop(
                            title
                        )
                        if st.session_state.current_chat == title:
                            st.session_state.current_chat = new_title
                        save_chat(new_title, st.session_state.chats[new_title])

                        extension = ("." * ("." not in title)) + "json"
                        os.remove(os.path.join(SAVE_FOLDER, f"{title}{extension}"))

                        st.rerun()


def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "dataframe" in message:
                df = pd.DataFrame(message["dataframe"])
                st.dataframe(df)


def process_prompt(prompt, client):
    """Process the user's prompt and handle different types of queries."""

    # Save prompt to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # print(st.session_state.messages)
    with st.chat_message("user"):
        st.write(prompt)

    # Check if the prompt is about retrieving a table
    # (i.e., if it's similar to "Get me a table of" in the beginning or
    # if it contains the phrase at all.)
    cleaned_prompt = prompt.strip().lower()
    words_in_list = cleaned_prompt.split()
    words_to_take = len(words_in_list) if len(words_in_list) < 5 else 5
    prompt_start = " ".join(cleaned_prompt.split()[:words_to_take])

    # Get the command embeddings to compare with
    with open(COMMAND_EMB) as f:
        embeddings = json.load(f)

    if "get me a table of" in cleaned_prompt or (
        compute_similarity(
            compute_embedding(prompt_start),
            embeddings[st.session_state["model_encoder"]],
        )
        > 0.5
    ):

        st.session_state.prompt = prompt

        # Get the table
        df = aggregate_tables(
            prompt,
            matching_threshold=st.session_state["matching_threshold"],
            tolerance=st.session_state["tolerance"],
            model_encoder=st.session_state["model_encoder"],
            temperature=st.session_state["temperature"],
        )

        if df is not None:

            st.session_state.prompt = ""

            table_msg = "Here is the table you requested:"
            st.session_state.last_dataframe = df
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": table_msg,
                    "dataframe": df.to_dict(),
                }
            )

            # Showing thee table to user
            with st.chat_message("assistant"):
                st.write(table_msg)
                st.dataframe(df)

    else:
        handle_text_based_query(prompt, client)


def handle_text_based_query(prompt, client):
    """Handle follow-up text-based queries that involve a DataFrame."""

    # Get the last DataFrame to ask a question about it
    df = st.session_state.get("last_dataframe", None)

    # Limit the number of messages to send to OpenAI
    window_hisotry = int(round(len(st.session_state.messages) * 0.75))
    messages_to_send = st.session_state.messages[-(window_hisotry * 2) :]
    prompt = messages_to_send

    if df is not None:
        response = process_dataframe_query(prompt, client, df)

        if isinstance(response, pd.DataFrame):
            st.session_state.last_dataframe = response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Here is the updated table based on your query:",
                    "dataframe": response.to_dict(),
                }
            )

            # Show the updated table to the user
            with st.chat_message("assistant"):
                st.dataframe(response)

        else:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

    else:
        stream_openai_response(client)


def stream_openai_response(client):
    """Stream OpenAI response for non-DataFrame queries."""
    stream = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
    )
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})


def auto_save_chat(client):
    """Auto-save and generate a title for new chats."""
    if (
        st.session_state.current_chat == "New Chat"
        and len(st.session_state.messages) == 2
    ):
        new_title = generate_title(client, st.session_state.messages)
        st.session_state.current_chat = new_title
        st.session_state.chats[new_title] = st.session_state.messages
        save_chat(new_title, st.session_state.messages)
        st.rerun()

    elif st.session_state.current_chat != "New Chat":
        st.session_state.chats[st.session_state.current_chat] = (
            st.session_state.messages
        )
        save_chat(st.session_state.current_chat, st.session_state.messages)


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
