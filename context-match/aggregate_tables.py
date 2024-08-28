import json
import argparse
import pandas as pd
import streamlit as st
from utils.constants import *
from utils.seed_initializer import set_seed
from utils.progress_bar import ProgressTracker
from utils.prompt_expansion import handle_prompt
from utils.supress_warnings import suppress_warnings
from utils.prompt_to_openai import get_relevant_columns
from utils.entry_matcher import enrich_dataframes, merge_top_k
from utils.table_saver import save_dataframe, check_merged_table
from utils.fetch_table_notion import get_dataframes, main_sort_dataframes


def aggregate_tables(
    prompt: str,
    matching_threshold: float = 0.5,
    tolerance: float = 0.1,
    model_encoder: str = "all-MiniLM-L6-v2",
    fetch_tables: bool = False,
    temperature: float = 0.0,
):

    # Initialize variables
    set_seed()
    suppress_warnings()
    progress = ProgressTracker(total_steps=5)
    arguments = {k: v for k, v in locals().items() if k != "prompt"}
    args = argparse.Namespace(**arguments)

    with open(KEYFILE_LOC) as f:
        dic_keys = json.load(f)
        NOTION_TOKEN = dic_keys["notion_token"]
        DATABASE_ID = dic_keys["database_id"]
        OPENAI_TOKEN = dic_keys["openAI_token"]

    # Stage 1: Start the process
    if st.session_state.process_stage == "start":

        # Prompt enrichment for refined search
        enriched_prompt = handle_prompt(
            prompt,
            api_key=OPENAI_TOKEN,
            print_prompt=True,
            expand_with_syn=False,
            expand_with_openAI=True,
        )

        # Update status
        progress.update("📥 Retrieving databases...")

        # Get the dataframes from Notion
        df_dict = get_dataframes(NOTION_TOKEN, DATABASE_ID, args)

        # Score each table based on similarity to the prompt
        df_ranked, df_fact_ranked = main_sort_dataframes(
            df_dict, enriched_prompt, OPENAI_TOKEN
        )

        st.session_state.df_ranked = df_ranked
        st.session_state.df_fact_ranked = df_fact_ranked
        st.session_state.process_stage = "user_selection"

    # Stage 2: User selection
    if st.session_state.process_stage == "user_selection":

        # Skip this step if there are less than 3 dataframes to select from
        if len(st.session_state.df_ranked) < 3:

            st.session_state.user_selection = st.session_state.df_ranked

        else:

            st.session_state.checkbox_values = {
                table_name: True for table_name in st.session_state.df_ranked.keys()
            }

            with st.form(key="table_selection_form"):
                st.write("Please select the tables you want to merge:")

                for table_name in st.session_state.df_ranked.keys():
                    st.session_state.checkbox_values[table_name] = st.checkbox(
                        label=table_name,
                        value=st.session_state.checkbox_values[table_name],
                    )

                submit_button = st.form_submit_button(label="Submit Selection")
                if submit_button:
                    st.session_state.user_selection = {
                        k: v for k, v in st.session_state.checkbox_values.items() if v
                    }
                    st.session_state.form_submitted = True
                    # st.rerun()

        st.session_state.process_stage = "continue_processing"

    # Stage 3: Continue processing
    if st.session_state.process_stage == "continue_processing":

        progress.update("🔍 Identifying relevant terms in the prompt...")

        # Skip this step if there is only one dataframe
        if len(st.session_state.df_ranked) > 1:
            # Filter df_ranked based on user selection
            df_ranked = {
                k: v
                for k, v in st.session_state.df_ranked.items()
                if k in st.session_state.user_selection
            }

            # Check if the dataframe cache already exists
            merged_df = check_merged_table(df_ranked)
            if merged_df is not None:

                progress.update("📧 Existing merge found! Retrieving...")
                final_df = merged_df

                # Clear the progress bar and any associated text
                progress.update("✅ Finalizing the table...")

                # Finalize for the progress bar
                progress.finalize()

                # Reset to the beginning
                st.session_state.process_stage = "start"

                return final_df

            # Else we continue
            else:
                dict_weights = get_relevant_columns(
                    prompt, df_ranked, OPENAI_TOKEN, args, verbose=True
                )

        st.session_state.process_stage = "add_context"

    ## Unused Danilo appreciation form (the answer has to be 'Yes' by the way)
    #
    #    # Streamlit form
    #     with st.form(key="danilo_form"):
    #         st.header("Before that answer this question: ")
    #         st.write("Do you like to work with Danilo? 👀")
    #         response = st.radio(
    #             label="Select your response:",
    #             options=["Yes", "No"],
    #         )

    #         submit_button = st.form_submit_button(label="Submit Response")
    #         if submit_button:
    #             # Store the user's response in session state
    #             st.session_state.user_selection = response
    #             st.session_state.process_stage = "continue_next_stage"
    #             if response == "Yes":
    #                 st.success(f"🥳 Your response '{response}' has been recorded. 🫡")
    #                 time.sleep(2)
    #             elif response == "No":
    #                 st.success(f"😰 Your response '{response}' has been recorded. 🫡")
    #                 time.sleep(2)

    # Stage 4: Add extra context to the tables
    if st.session_state.process_stage == "add_context":

        progress.update("🔗 Adding additional context to the tables found...")

        # Enrich the dataframes with Fact tables
        df_enriched = enrich_dataframes(df_ranked, st.session_state.df_fact_ranked)

        # Immediately return the dataframe if there's only 1
        if len(st.session_state.df_ranked) < 2:

            final_df = st.session_state.df_ranked[
                next(iter(st.session_state.df_ranked))
            ]
            progress.update("✅ Finished processing!")

        else:
            progress.update(
                "🛠️ Merging tables...",
                "<span style='color:gray; font-size:0.9em;'>**Note:** this may take a while depending on the size of the tables.</span>",
            )

            # Merge the enriched dataframes
            final_df = merge_top_k(
                prompt, df_enriched, dict_weights, OPENAI_TOKEN, args
            )

            # Clear the progress bar and any associated text
            progress.update("✅ Finalizing the table...")

        # Finalize for the progress bar
        progress.finalize()

        # Cleanup the final table
        final_df.dropna(inplace=True)

        # Write the dataframe to the cache table and save it if not already present
        save_dataframe(final_df, st.session_state.df_ranked)

        # Reset the process stage for the next run
        st.session_state.process_stage = "start"

        return final_df

    return None


if __name__ == "__main__":

    table = aggregate_tables("Get me a table of firms")
    if table is not None:
        table.to_csv("data/final.csv")
