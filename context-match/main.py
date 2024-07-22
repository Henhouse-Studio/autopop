import json
import argparse
from notion_client import Client
from utils.filter_names import *
from utils.make_embeddings import *
from utils.prompt_expansion import *
from utils.fetch_table_notion import *
from utils.compute_similarity import *


# Argparser:
def config():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", default=0.8, type=float, help="Threshold for detecting matches"
    )

    return parser.parse_args()


# Entry matching
def entry_matcher(
    df_base: pd.DataFrame, df_populate: pd.DataFrame, filtered_similarity_scores: dict
):
    print("Matching the entries...")
    merged_data = []

    # Iterate over the pointers to merge the rows
    pointers = list(filtered_similarity_scores.keys())
    for i, j in pointers:

        # Get the row from df_base
        df_base_row = df_base.iloc[i].copy()
        # Get the row from df_populate and rename its columns
        df_populate_row = df_populate.iloc[j].copy()
        df_populate_row.index = [f"{col}_df2" for col in df_populate_row.index]

        # Concatenate the rows along the columns
        merged_row = pd.concat([df_base_row, df_populate_row])
        merged_data.append(merged_row)

    # Convert the list of merged rows to a DataFrame
    result_df = pd.DataFrame(merged_data)
    print("Matching done!\n")

    return result_df


# Execution
if __name__ == "__main__":

    # Get the argparser arguments
    args = config()

    # Initialize the Notion client
    with open("notion.json") as f:
        NOTION_TOKEN = json.load(f)["KEY"]

    notion = Client(auth=NOTION_TOKEN)

    # The Notion database ID
    database_id = "6ead038babe946b99854dba84ecf05a9"

    # Get the page and table links from the database
    page_names, page_links = get_page_links(notion, database_id)
    page_table_links = get_table_links_from_pages(notion, page_links)

    # Prompt from the user
    prompt = "Get me a table of firms and their employees"
    # prompt = "Get me a table of people's job profiles"

    # Enrichment of the prompt
    prompt = expand_prompt_with_synonyms(prompt)
    # print(prompt)
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

    final_df = entry_matcher(df_first, df_second, filtered_similarity_scores)
    final_df = final_df.drop(f"{highest_similar_col_name}_df2", axis="columns")

    # Remove columns which are the same
    final_df = remove_duplicates(final_df)

    final_df.to_csv("out.csv", index=False)
    # print(final_df)

    print("Dataset exported!")
