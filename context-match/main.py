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
        "--threshold", default=0.7, type=float, help="Threshold for detecting matches"
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
    # prompt = "Get me a table of people's job profiles"

    # Enrichment of the prompt
    prompt = expand_prompt_with_synonyms(prompt)
    # print(prompt)
    prompt_embedding = compute_embedding(prompt)

    dfs_dic = get_dataframes(page_table_links, page_names, NOTION_TOKEN)

    # Converting the databases to pandas dataframes
    dfs_dict_ranked, len_grouped_data  = score_dataframes(dfs_dic, prompt_embedding)

    # Get the top-k similar dataframes
    df_ranked = list(dfs_dict_ranked.items())[:len_grouped_data]

    # printing similarity score, name of df_ranked
    print('\nTop-k similar dataframes:')
    for i in range(len_grouped_data):
        print(f"{df_ranked[i][1][0]} {df_ranked[i][0]}")

    # Selecting the first two dataframes for comparison
    if len(df_ranked) < 2:
        # print("Not enough dataframes to compare!")
        df_second = pd.DataFrame()
    else: 
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
