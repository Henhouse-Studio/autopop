import json
import streamlit as st
from utils.constants import *


def display_welcome_message():
    st.markdown(
        """
        ## Welcome to the AutoPop ChatBot!
        
        I am your assistant, here to help you with various tasks:
        
        - **Fetch and aggregate tables**: Just ask me to get a table of something!
        - **General Questions**: Ask me anything else, and I'll try my best to assist.
        
        **How to use:**
        - Type your request in the input box below.
        - You can ask for a table by starting with "Get me a table of...".
        - Or just chat with me to get started!
        
        ### What would you like to do today?
        """
    )
    if st.button("Get a table"):
        prompt = "Get me a table of "
        return prompt  # Pre-fill with a table request

    return None


def process_dataframe_query(prompt, client, df):
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


def save_chat(title, messages):

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    file_path = os.path.join(SAVE_FOLDER, f"{title}.json")

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
                            f"Warning: Skipping file '{filename}' due to JSONDecodeError."
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

    summary_prompt = f"Summarize the following chat in 5 words or less:\n\n{messages[0]['content']}\n{messages[1]['content']}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


def load_css(file_name):

    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
