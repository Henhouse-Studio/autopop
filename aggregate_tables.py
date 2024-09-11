import json
import argparse
import pandas as pd
import streamlit as st
from utils.constants import *
from utils.seed_initializer import set_seed
from utils.prompt_expansion import handle_prompt
from utils.supress_warnings import suppress_warnings
from utils.prompt_to_openai import get_relevant_columns
from utils.entry_matcher import enrich_dataframes, merge_top_k
from utils.table_saver import save_dataframe, check_merged_table
from utils.progress_history import ProgressTracker, HistoryTracker
from utils.fetch_table_notion import get_dataframes, main_sort_dataframes


def initialize_environment(args):

    set_seed()
    suppress_warnings()
    progress = ProgressTracker(total_steps=5)
    history = HistoryTracker()

    # with open(KEYFILE_LOC) as f:
    #     dic_keys = json.load(f)
    #     NOTION_TOKEN = dic_keys["notion_token"]
    #     DATABASE_ID = dic_keys["database_id"]
    #     OPENAI_TOKEN = dic_keys["openAI_token"]

    NOTION_TOKEN = st.secrets["notion_token"]
    DATABASE_ID = st.session_state["database_ID"]
    OPENAI_TOKEN = st.secrets["openAI_token"]

    return progress, history, NOTION_TOKEN, DATABASE_ID, OPENAI_TOKEN


def handle_start_stage(
    prompt, history, OPENAI_TOKEN, NOTION_TOKEN, DATABASE_ID, args, progress
):

    enriched_prompt = handle_prompt(
        prompt,
        api_key=OPENAI_TOKEN,
        verbose=False,
        expand_with_syn=False,
        expand_with_openAI=True,
    )

    progress.update("📥 Retrieving databases...")
    df_dict = get_dataframes(NOTION_TOKEN, DATABASE_ID, args)
    history.update()

    df_ranked, df_fact_ranked = main_sort_dataframes(
        df_dict, enriched_prompt, OPENAI_TOKEN
    )
    history.update()

    st.session_state.df_ranked = df_ranked
    st.session_state.df_fact_ranked = df_fact_ranked
    st.session_state.process_stage = "user_selection"


def handle_user_selection_stage():

    st.session_state.checkbox_values = {
        table_name: True for table_name in st.session_state.df_ranked.keys()
    }

    if "show_form" not in st.session_state:
        st.session_state.show_form = True

    if st.session_state.show_form:

        # The form for selecting which databases to include
        with st.form(key="table_selection_form"):

            st.write("Please select the tables you want to merge:")
            for table_name in st.session_state.df_ranked.keys():
                st.session_state.checkbox_values[table_name] = st.checkbox(
                    label=table_name,
                    value=st.session_state.checkbox_values.get(table_name, True),
                )

            submit_button = st.form_submit_button(label="Submit Selection")
            if submit_button:
                selected_tables = {
                    k: v for k, v in st.session_state.checkbox_values.items() if v
                }

                if selected_tables:
                    st.session_state.user_selection = selected_tables
                    st.session_state.form_submitted = True
                    st.session_state.process_stage = "continue_processing"
                    st.session_state.show_form = False  # Hide the form after submission
                    st.rerun()  # Force a rerun to update the UI immediately

                else:
                    st.warning("You must select at least one table to proceed.")

    else:
        st.write("Table selection completed. Processing...")

    # Reset the form visibility when returning to the start stage
    if st.session_state.process_stage == "start":
        st.session_state.show_form = True


def handle_continue_processing_stage(prompt, history, OPENAI_TOKEN, args, progress):

    progress.update("🔍 Identifying relevant terms in the prompt...")

    df_ranked = {
        k: v
        for k, v in st.session_state.df_ranked.items()
        if k in st.session_state.user_selection
    }
    st.session_state.df_ranked = df_ranked

    merged_df = check_merged_table(df_ranked)
    history.update()

    if merged_df is not None:
        st.write("Returning cached table...")
        progress.update("📧 Existing merge found! Retrieving...")
        progress.update("✅ Finalizing the table...")
        progress.finalize()
        st.session_state.process_stage = "start"

        return merged_df

    else:
        dict_weights = get_relevant_columns(
            prompt, df_ranked, OPENAI_TOKEN, args, verbose=False
        )
        history.update()
        st.session_state.process_stage = "add_context"

        return dict_weights


def handle_add_context_stage(
    prompt, history, OPENAI_TOKEN, args, progress, dict_weights
):

    progress.update("🔗 Adding additional context to the tables found...")
    df_enriched = enrich_dataframes(
        st.session_state.df_ranked, st.session_state.df_fact_ranked, verbose=False
    )
    history.update()

    # if len(st.session_state.df_ranked) < 2:
    #     final_df = st.session_state.df_ranked[next(iter(st.session_state.df_ranked))][0]
    #     progress.update("✅ Finished processing!")

    # else:
    progress.update(
        "🛠️ Merging tables...",
        "<span style='color:gray; font-size:0.9em;'>**Note:** this may take a while depending on the size of the tables.</span>",
    )
    final_df = merge_top_k(prompt, df_enriched, dict_weights, OPENAI_TOKEN, args)
    history.update()
    progress.update("✅ Finalizing the table...")

    # Clean up the result
    final_df.dropna(inplace=True)

    # Save it if there's more than one dataframe involved:
    if len(st.session_state.df_ranked) > 1:
        save_dataframe(final_df, st.session_state.df_ranked)

    progress.finalize()
    history.finalize()
    st.session_state.process_stage = "start"

    return final_df


def aggregate_tables(
    prompt: str,
    matching_threshold: float = 0.5,
    tolerance: float = 0.1,
    model_encoder: str = "all-MiniLM-L6-v2",
    fetch_tables: bool = True,
    temperature: float = 0.0,
):
    # # Set progress bar running state to True
    # st.session_state.progress_running = True

    arguments = {k: v for k, v in locals().items() if k != "prompt"}
    args = argparse.Namespace(**arguments)

    progress, history, NOTION_TOKEN, DATABASE_ID, OPENAI_TOKEN = initialize_environment(
        args
    )

    if st.session_state.process_stage == "start":

        # For the form
        st.session_state.show_form = True
        handle_start_stage(
            prompt, history, OPENAI_TOKEN, NOTION_TOKEN, DATABASE_ID, args, progress
        )

    if st.session_state.process_stage == "user_selection":
        handle_user_selection_stage()

    if st.session_state.process_stage == "continue_processing":
        result = handle_continue_processing_stage(
            prompt, history, OPENAI_TOKEN, args, progress
        )

        # Adjust the flow depending on the results
        if isinstance(result, pd.DataFrame):
            return result

    if st.session_state.process_stage == "add_context":

        final = handle_add_context_stage(
            prompt, history, OPENAI_TOKEN, args, progress, result
        )

        return final

    return None


if __name__ == "__main__":

    table = aggregate_tables("Get me a table of firms")
    if table is not None:
        table.to_csv("data/final.csv")
