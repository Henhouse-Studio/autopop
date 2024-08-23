import sys
import json
import argparse

# from utils.fuzzy_matcher import *
from utils.make_embeddings import *
from utils.prompt_expansion import *
from utils.fetch_table_notion import *
from utils.compute_similarity import *
from utils.entry_matcher import *
from utils.prompt_to_openai import *
from utils.supress_warnings import *
from utils.seed_initializer import *


# Argparser:
def config():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matching_threshold",
        default=0.5,
        type=float,
        help="Threshold required to merge two DataFrames",
    )
    parser.add_argument(
        "--tolerance",
        default=0.1,
        type=float,
        help="Controls when two rows are allowed to match",
    )
    parser.add_argument(
        "--model_encoder",
        default="all-MiniLM-L6-v2",
        type=str,
        help="Model to encode the text",
    )
    parser.add_argument(
        "--fetch_tables",
        action="store_false",
        help="Whether to fetch tables from Notion",
    )
    parser.add_argument(
        "--temperature",
        default=0.0,
        type=float,
        help="Controls randomness in API responses; 1.0 - more varied outputs",
    )

    return parser.parse_args()


# Execution
if __name__ == "__main__":

    # Set the seed for reproducibility
    set_seed()

    # Suppress warnings and logging
    suppress_warnings()

    # Get the argparse arguments
    args = config()

    # Get the required keys from the JSON file
    with open("keys.json") as f:
        dic_keys = json.load(f)
        NOTION_TOKEN = dic_keys["notion_token"]
        DATABASE_ID = dic_keys["database_id"]
        OPENAI_TOKEN = dic_keys["openAI_token"]

    # The user prompt
    # prompt = "Get me a table of firms and their employees"
    # prompt = "Get me a table of blogpost authors and their LinkedIn profiles"
    # prompt = "I want to find people who work for the government"
    prompt = "Get me a table of blogpost authors and their companies"
    # prompt = "Get me tables that contain people and their companies"
    # prompt = "Get me a table that contains people"
    # prompt = "Get me a table that contains only companies "
    # prompt = "Get me a table that contains only skills of people"

    # Prompt enrichment for refined search
    enriched_prompt = handle_prompt(
        prompt,
        api_key=OPENAI_TOKEN,
        print_prompt=True,
        expand_with_syn=False,
        expand_with_openAI=True,
    )
    # prompt_embedding = compute_embedding(enriched_prompt)

    # Get the dataframes from Notion
    df_dict = get_dataframes(NOTION_TOKEN, DATABASE_ID, args)

    # Score each table how similar it is to the prompt
    df_ranked, df_fact_ranked = main_sort_dataframes(
        df_dict, enriched_prompt, OPENAI_TOKEN
    )

    dict_weights = get_relevant_columns(
        prompt, df_ranked, OPENAI_TOKEN, args, verbose=True
    )

    # Enrich the dataframes with Fact tables
    df_enriched = enrich_dataframes(df_ranked, df_fact_ranked)

    # Merge the enriched dataframes
    final_df = merge_top_k(prompt, df_enriched, dict_weights, OPENAI_TOKEN, args)
    final_df.to_csv("final.csv", index=False)

    print("Dataset exported!")
