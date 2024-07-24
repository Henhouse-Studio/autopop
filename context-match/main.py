import json
import argparse
from notion_client import Client
from utils.filter_names import *
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
        "--threshold", default=0.8, type=float, help="Threshold for detecting matches"
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
    notion = Client(auth=NOTION_TOKEN)

    # Get the page and table links from the database
    page_names, page_links = get_page_links(notion, DATABASE_ID)
    page_table_links = get_table_links_from_pages(notion, page_links)

    # Prompt from the user
    prompt = "Get me a table of firms and their employees"
    # prompt = "Get me a table of people's job profiles"

    # Enrichment of the prompt
    prompt = expand_prompt_with_synonyms(prompt)
    print(prompt)
    prompt_embedding = compute_embedding(prompt)

    # Converting the databases to pandas dataframes
    df_dict = to_pandas(page_table_links, page_names, prompt_embedding, NOTION_TOKEN)

    # Similarity scores between all rows in both databases
    df_ranked = list(df_dict.items())
    df_first = df_ranked[0][1][1]
    df_second = df_ranked[1][1][1]
    score_dict, highest_similar_col_name = compute_similarity_softmax(
        df_first, df_second
    )

    # Threshold based on table size
    threshold = 2 * args.threshold / len(df_second)

    # Filter similarity scores based on threshold
    filtered_similarity_scores = {k: v for k, v in score_dict.items() if v >= threshold}

    print(f"Found {len(filtered_similarity_scores)} matches!\n")

    final_df = combine_dfs(df_first, df_second, filtered_similarity_scores)
    # final_df = final_df.drop(f"{highest_similar_col_name}_df2", axis="columns")

    # Remove columns which are the same
    final_df = remove_duplicates(final_df)

    # Rename columns in case there are similar names
    final_df = rename_columns(final_df, api_key=OPENAI_TOKEN)

    final_df.to_csv("out.csv", index=False)
    # print(final_df)

    print("Dataset exported!")
