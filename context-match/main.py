import json
import argparse
from notion_client import Client

# from utils.fuzzy_matcher import *
from utils.make_embeddings import *
from utils.prompt_expansion import *
from utils.fetch_table_notion import *
from utils.compute_similarity import *
from utils.entry_matcher import *
from utils.prompt_to_openai import *


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
    return parser.parse_args()


# Execution
if __name__ == "__main__":

    # Get the argparse arguments
    args = config()

    # get keys from json key holder
    with open("keys.json") as f:
        dic_keys = json.load(f)
        NOTION_TOKEN = dic_keys["notion_token"]
        DATABASE_ID = dic_keys["database_id"]
        OPENAI_TOKEN = dic_keys["openAI_token"]

    # Initialize the Notion client
    notion_client = Client(auth=NOTION_TOKEN)

    # Get the page and table links from the database
    page_names, page_links = get_page_links(notion_client, DATABASE_ID)
    page_table_links = get_table_links_from_pages(notion_client, page_links)

    # Prompt from the user
    prompt = "Get me a table of firms and their employees"
    # prompt = "Get me a table of employees and their job profiles"
    # prompt = "Get me a table of employees, their skills and their job profiles"
    # prompt = "Get me a table of employees"
    # prompt = "Get me a table of people's job profiles"

    # Enrichment of the prompt
    prompt = expand_prompt_with_synonyms(prompt)
    prompt = get_enriched_prompt(prompt, api_key=OPENAI_TOKEN)
    # print(prompt)
    prompt_embedding = compute_embedding(prompt)

    df_dict = get_dataframes(page_table_links, page_names, NOTION_TOKEN)

    # Converting the databases to pandas dataframes
    df_ranked, df_fact_ranked = score_dataframes(df_dict, prompt_embedding)

    df_enriched = enrich_dataframes(df_ranked, df_fact_ranked)

    # df_enriched["LinkedIn Profiles"].to_csv("enriched.csv", index=False)

    final_df = merge_top_k(df_enriched, OPENAI_TOKEN, args)

    # # print(final_df)

    final_df.to_csv("final.csv")

    print("Dataset exported!")
