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
from utils.verbosity import *


# Argparser:
def config():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", default=0.7, type=float, help="Threshold for detecting matches"
    )
    parser.add_argument(
        "--model_encoder",
        default="all-MiniLM-L6-v2",
        type=str,
        help="Model to encode the text",
    )
    parser.add_argument(
        "--load_local_tables",
        default=True,
        type=bool,
        help="Load saved csv tables from local storage",
    )
    return parser.parse_args()


# Execution
if __name__ == "__main__":

    # Suppress warnings and logging
    suppress_warnings()

    # Get the argparse arguments
    args = config()

    # Get keys from json key holder
    with open("keys.json") as f:
        dic_keys = json.load(f)
        NOTION_TOKEN = dic_keys["notion_token"]
        DATABASE_ID = dic_keys["database_id"]
        OPENAI_TOKEN = dic_keys["openAI_token"]

    # Prompt from the user
    prompt = "Get me a table of firms and their employees"
    # prompt = "Get me a table of employees and their job profiles"

    # Enrichment of the prompt
    enriched_prompt = handle_prompt(
        prompt,
        api_key=OPENAI_TOKEN,
        print_prompt=True,
        expand_with_syn=True,
        expand_with_openAI=True,
    )
    prompt_embedding = compute_embedding(enriched_prompt)

    # compute the embeddings of the fields for every table
    # df_fields = pd.read_csv("databases/table_of_fields/table_of_fields.csv")
    # df_fields["embedding"] = df_fields["Description"].apply(compute_embedding)
    # df_fields.to_csv("databases/table_of_fields/table_of_fields.csv", index=False)
    # df_fields_ranked = score_fields(df_fields, prompt_embedding)

    # Get the dataframes from Notion
    df_dict = get_dataframes(NOTION_TOKEN, DATABASE_ID, args)

    # Score each table how similar it is to the prompt
    df_ranked, df_fact_ranked = score_dataframes(df_dict, prompt_embedding)

    dict_weightss = get_relevant_columns(prompt, df_ranked, OPENAI_TOKEN)

    # Enrich the dataframes with Fact tables
    df_enriched = enrich_dataframes(df_ranked, df_fact_ranked)
    # df_enriched["LinkedIn Profiles"].to_csv("enriched.csv", index=False)

    # Merge the enriched dataframes
    final_df = merge_top_k(df_enriched, dict_weightss, OPENAI_TOKEN, args)
    final_df.to_csv("final.csv")

    print("Dataset exported!")
