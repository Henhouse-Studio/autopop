import json
import streamlit as st
from utils.constants import *


def load_css(file_name):

    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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
                        print(f"Warning: Skipping file '{filename}' due to JSONDecodeError.")
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
