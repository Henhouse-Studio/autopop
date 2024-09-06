# utils/progress_utils.py
import streamlit as st


class ProgressTracker:
    def __init__(self, total_steps: int = 5):
        """
        Initializes the progress bar and placeholders.

        :param total_steps: The total number of steps in the process (int).
        """

        self.result_holder = st.empty()
        self.note_placeholder = st.empty()
        self.current_step = 0
        self.total_steps = total_steps

    def update(self, step_description, note=""):
        """
        Updates the progress bar and displays the current progress status.

        :param step_description: The description of the current step (str).
        :param note: An optional note to display under the progress (str).
        """

        self.current_step += 1
        progress = self.current_step / self.total_steps
        with self.result_holder.container():
            st.progress(progress, f"Progress: {progress*100:.2f}%")
            st.markdown(step_description)

        if note:
            self.note_placeholder.markdown(note, unsafe_allow_html=True)

        else:
            self.note_placeholder.empty()

    def finalize(self):
        """Clears the progress bar and placeholders."""

        self.result_holder.empty()
        self.note_placeholder.empty()


class HistoryTracker:
    def __init__(self):
        """
        Initializes the progress bar and placeholders.
        """

        self.process_history = {}
        self.result_holder = st.empty()

    def update(self):
        """
        Updates the progress bar and displays the current progress status.
        """

        self.process_history = st.session_state.process_history
        content = next(reversed(self.process_history.values()))

        with self.result_holder.container():
            with st.expander("Show process"):
                st.write("Here's the output associated with this operation:")
                code_content = content
                st.code(code_content, language="python")

    def finalize(self):
        """Clears the progress bar and placeholders."""

        self.process_history = {}
        self.result_holder.empty()


def save_progress_text(text: str, verbose: bool = True):

    if verbose:
        print(text + "\n")

    if st.session_state.process_stage not in st.session_state.process_history:
        st.session_state.process_history[st.session_state.process_stage] = text

    else:
        st.session_state.process_history[st.session_state.process_stage] += text + "\n"
