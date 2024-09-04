import streamlit as st
import random


def get_random_code_snippet():
    snippets = [
        """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
        """,
        """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
        """,
        """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        """,
    ]
    return random.choice(snippets)


def main():
    st.set_page_config(layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_code" not in st.session_state:
        st.session_state.show_code = True

    if "current_code" not in st.session_state:
        st.session_state.current_code = get_random_code_snippet()

    # Sidebar with toggle button
    with st.sidebar:
        st.title("Controls")
        toggle_code = st.button("Toggle Code Column", key="toggle_code")
        if toggle_code:
            st.session_state.show_code = not st.session_state.show_code
            st.rerun()

    # Main content
    if st.session_state.show_code:
        col1, col2 = st.columns([3, 1])
    else:
        col1 = st.container()
        col2 = None

    with col1:
        st.title("Chat Interface")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What's up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = f"You said: {prompt}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    if st.session_state.show_code and col2:
        with col2:
            st.title("Code Display")
            st.code(st.session_state.current_code, language="python")
            if st.button("New Code Snippet"):
                st.session_state.current_code = get_random_code_snippet()


if __name__ == "__main__":
    main()
