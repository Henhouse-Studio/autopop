import json
import argparse
import pandas as pd
from utils.constants import *
from utils.seed_initializer import set_seed
from utils.prompt_expansion import handle_prompt
from utils.supress_warnings import suppress_warnings
from utils.prompt_to_openai import get_relevant_columns
from utils.entry_matcher import enrich_dataframes, merge_top_k
from utils.fetch_table_notion import get_dataframes, score_dataframes


# Execution
def aggregate_tables(
    prompt: str,
    matching_threshold: float = 0.5,
    tolerance: float = 0.1,
    model_encoder: str = "all-MiniLM-L6-v2",
    fetch_tables: bool = False,
    temperature: float = 0.0,
):
    # # load csv to a dataframe
    # df = pd.read_csv("final_combined_df.csv")

    # # drop the rows that have missing values from the df
    # df.dropna(inplace=True)

    # # return the dataframe
    # return df

    # return print("Your typed prompt is: ", prompt)
    # Collect all arguments into a dictionary, except for the 'prompt'
    arguments = {k: v for k, v in locals().items() if k != "prompt"}
    args = argparse.Namespace(**arguments)

    # Set the seed for reproducibility
    set_seed()

    # Suppress warnings and logging
    suppress_warnings()

    # Get the required keys from the JSON file
    # TODO: Make key loading more centralized
    with open(KEYFILE_LOC) as f:
        dic_keys = json.load(f)
        NOTION_TOKEN = dic_keys["notion_token"]
        DATABASE_ID = dic_keys["database_id"]
        OPENAI_TOKEN = dic_keys["openAI_token"]

    # Prompt enrichment for refined search
    enriched_prompt = handle_prompt(
        prompt,
        api_key=OPENAI_TOKEN,
        print_prompt=True,
        expand_with_syn=False,
        expand_with_openAI=True,
    )

    # Get the dataframes from Notion
    df_dict = get_dataframes(NOTION_TOKEN, DATABASE_ID, args)

    # Score each table how similar it is to the prompt
    df_ranked, df_fact_ranked = score_dataframes(df_dict, enriched_prompt, OPENAI_TOKEN)

    dict_weights = get_relevant_columns(
        prompt, df_ranked, OPENAI_TOKEN, args, verbose=True
    )

    # Enrich the dataframes with Fact tables
    df_enriched = enrich_dataframes(df_ranked, df_fact_ranked)

    # Merge the enriched dataframes
    final_df = merge_top_k(prompt, df_enriched, dict_weights, OPENAI_TOKEN, args)

    # Cleanup the final table
    final_df.dropna(inplace=True)

    return final_df


if __name__ == "__main__":

    table = aggregate_tables("Get me a table of firms")
    table.to_csv("data/final.csv")
