import os
import pickle
import notion_df
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from notion_client import Client
from utils.make_embeddings import *
from utils.compute_similarity import *
from utils.prompt_to_openai import *


def get_table_notion(NOTION_TOKEN, DATABASE_URL):
    """
    Retrieve a table from Notion as a DataFrame.

    :param NOTION_TOKEN: The Notion API token.
    :param DATABASE_URL: The URL of the Notion database.
    :return: A pandas DataFrame containing the data from the Notion database.
    """

    df = notion_df.download(DATABASE_URL, api_key=NOTION_TOKEN)
    df = df.iloc[:, ::-1]  # Reverse the columns for specific ordering
    return df


def get_page_links(notion_client, database_id):
    """
    Get links to pages in a Notion database.

    :param notion_client: The Notion client object.
    :param database_id: The ID of the Notion database.
    :return: A tuple containing a list of (flag_tag, page_title) and a list of page URLs.
    """

    response = notion_client.databases.query(database_id=database_id)
    pages = response.get("results")

    names = []
    links = []
    for page in pages:

        flag_tag = False
        page_title = page["properties"]["Name"]["title"][0]["text"]["content"]
        page_tags = page["properties"]["Tags"]["multi_select"]
        for tag in page_tags:

            if tag["name"] == "Fact":
                flag_tag = True
                break

        page_id = page["id"]
        page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
        names.append((flag_tag, page_title))
        links.append(page_url)

    return names, links


def get_table_links_from_pages(notion, page_links):
    """
    Get links to tables in each Notion page.

    :param notion: The Notion client object.
    :param page_links: A list of links to the Notion pages.
    :return: A dictionary with page links as keys and list of table links as values.
    """

    page_table_links = {}

    for page_link in page_links:
        page_id = page_link.split("/")[-1]
        blocks = notion.blocks.children.list(block_id=page_id).get("results")

        table_links = []
        for block in blocks:

            if block["type"] == "child_database":
                table_id = block["id"]
                table_url = f"https://www.notion.so/{page_id.replace('-', '')}#{table_id.replace('-', '')}"
                table_links.append(table_url)

        page_table_links[page_link] = table_links

    return page_table_links


def save_data_pickle(
    page_names,
    page_table_links,
    page_names_file="page_names.pkl",
    page_table_links_file="page_table_links.pkl",
):
    with open(page_names_file, "wb") as f:
        pickle.dump(page_names, f)

    with open(page_table_links_file, "wb") as f:
        pickle.dump(page_table_links, f)

    print(f"Data saved to {page_names_file} and {page_table_links_file}")


def load_data_pickle(
    page_names_file="page_names.pkl", page_table_links_file="page_table_links.pkl"
):
    with open(page_names_file, "rb") as f:
        page_names = pickle.load(f)

    with open(page_table_links_file, "rb") as f:
        page_table_links = pickle.load(f)

    print(f"Data loaded from {page_names_file} and {page_table_links_file}")
    return page_names, page_table_links


def get_dataframes(notion_token: str, database_id: str, args: argparse.Namespace):
    """
    Retrieve dataframes from Notion pages and store them locally if not already saved.

    :param notion_token: The Notion API token.
    :param database_id: The ID of the Notion database.
    :param args: Argparser namespace containing the parameter 'fetch_tables'.
    :return: A dictionary with table names as keys and pandas DataFrames as values.
    """
    print("Retrieving databases...")

    page_names_file = "databases/table_of_tables/page_names.pkl"
    page_table_links_file = "databases/table_of_tables/page_table_links.pkl"

    if (
        os.path.exists(page_names_file)
        and os.path.exists(page_table_links_file)
        and args.fetch_tables
    ):
        # Load saved data
        page_names, page_table_links = load_data_pickle(
            page_names_file, page_table_links_file
        )

    else:
        # Initialize the Notion client
        notion_client = Client(auth=notion_token)

        # Get the page and table links from the database
        page_names, page_links = get_page_links(notion_client, database_id)
        page_table_links = get_table_links_from_pages(notion_client, page_links)

        # Save the data
        save_data_pickle(
            page_names, page_table_links, page_names_file, page_table_links_file
        )

    df_dict = {}
    for idx, tables in enumerate(page_table_links.values()):

        is_fact = page_names[idx][0]
        table_name = page_names[idx][1]
        path_table = f"databases/table_of_tables/{table_name}.csv"
        print(f"[{idx+1}] Found{' Fact' * is_fact} table: '{table_name}'")

        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"

            if is_fact:

                if os.path.isfile(path_table) and not args.fetch_tables:

                    df = pd.read_csv(path_table)

                else:
                    print(
                        "Saving Fact table. This may take a while as it is a large database..."
                    )
                    df = get_table_notion(notion_token, temp)
                    df.to_csv(path_table, index=False)

            else:

                if os.path.isfile(path_table) and not args.fetch_tables:
                    df = pd.read_csv(path_table)

                else:
                    df = get_table_notion(notion_token, temp)
                    df.to_csv(path_table, index=False)

            df_dict[table_name] = (is_fact, df)

    print(f"[*] Number of databases found: {len(df_dict)}\n")
    return df_dict


def score_dataframes(dfs_dict: dict, enriched_prompt: str, openai_token: str):
    """
    Score each dataframe based on similarity to a prompt embedding.

    :param dfs_dict: A dictionary with table names as keys and pandas DataFrames as values (dict).
    :param enriched_prompt: The enriched prompt text (str).
    :param openai_token: The token to access ChatGPT (str).
    :return: A tuple containing the sorted and fact dataframes (dict, dict).
    """
    print("Scoring databases based on prompt...")

    df_dict = {}
    df_fact_dict = {}
    for table_name, (is_fact, df) in dfs_dict.items():

        col_names = list(df.columns)
        sample = df.sample(n=2, random_state=42)
        desc = f"The name of the table is {table_name}. This is a extract from the table:\n"

        for sample_value in range(len(sample)):
            for col_name in col_names:

                desc += f"{col_name}: {sample[col_name].values[sample_value]}\n"

        # # print(desc)
        # field_embeddings = compute_embedding(desc)
        # similarity_score = compute_similarity(prompt_embedding, field_embeddings)
        # similarity_score = round(similarity_score * 100, 2)
        similarity_score = 0

        if is_fact:
            df_fact_dict[table_name] = (similarity_score, df, desc)

        else:
            df_dict[table_name] = (similarity_score, df, desc)

    # Sorting the dictionary based on similarity score
    df_dict = dict(sorted(df_dict.items(), key=lambda x: x[1][0], reverse=True))
    df_fact_dict = dict(
        sorted(df_fact_dict.items(), key=lambda x: x[1][0], reverse=True)
    )

    # Names of the relevant tables based on prompt
    relevant_tables = rerank_dataframes(enriched_prompt, df_dict, openai_token)
    relevant_fact_tables = rerank_dataframes(
        enriched_prompt, df_fact_dict, openai_token
    )

    # Filter out the relevant tables from df_dict by the names
    # df_ranked = {
    #     table_name: value
    #     for table_name, value in df_dict.items()
    #     if table_name in relevant_tables
    # }

    # df_fact_ranked = {
    #     table_name: value
    #     for table_name, value in df_fact_dict.items()
    #     if table_name in relevant_fact_tables
    # }

    df_ranked = {}
    for table_name in relevant_tables:
        df_ranked[table_name] = df_dict[table_name]

    df_fact_ranked = {}
    for table_name in relevant_fact_tables:
        df_fact_ranked[table_name] = df_fact_dict[table_name]

    print(f"Selecting Top-{len(relevant_tables)} tables:")

    # printing similarity score, name of df_ranked
    for i, (key, _) in enumerate(df_ranked.items()):

        print(f"[{i+1}]:", key)

    print(f"Selecting Top-{len(relevant_fact_tables)} Fact tables:")

    # printing similarity score, name of df_ranked
    for i, (key, _) in enumerate(df_fact_ranked.items()):

        print(f"[{i+1}]:", key)

    return df_ranked, df_fact_ranked


# Currently Unused
def get_top_k(dfs_dict_ranked: dict):
    """
    Get the top-k similar dataframes based on the similarity scores.

    :param dfs_dict_ranked: A dictionary of dataframes ranked by similarity score (dict).
    :return: The number of top-k similar dataframes (int).
    """

    data = [df[0] for df in dfs_dict_ranked.values()]
    std_dev = np.std(data)

    current_group = [data[0]]
    top_k = 1

    for i in range(1, len(data)):

        if abs(data[i] - np.mean(current_group)) <= std_dev:
            current_group.append(data[i])
            top_k += 1

        else:
            break

    return top_k


# Currently Unused
def score_fields(dfs_dict: dict, prompt_embedding: np.array):
    """
    Score each dataframe based on similarity to a prompt embedding.

    :param dfs_dict: A dictionary with table names as keys and pandas DataFrames as values (dict).
    :param prompt_embedding: A numpy array representing the embedding of the prompt (np.array).
    :return: A tuple containing the sorted and fact dataframes (dict, dict).
    """
    print("Scoring table of fields based on prompt...")

    # iterate over each row from the dfs_dict
    for index, row in dfs_dict.iterrows():

        # compute similarity score with prompt embedding
        similarity_score = compute_similarity(row["embedding"], prompt_embedding)
        similarity_score = round(similarity_score * 100, 2)

        # append the similarity score to the dataframe
        dfs_dict.at[index, "similarity_score"] = similarity_score

    dfs_dict.to_csv("databases/table_of_fields/table_of_fields.csv", index=False)

    return dfs_dict


# Currently Unused
def remove_duplicates(df: pd.DataFrame, threshold: float = 0.9):
    """
    Remove duplicate columns from a dataframe based on similarity threshold.

    :param df: The pandas DataFrame to process.
    :param threshold: The similarity threshold for considering columns as duplicates.
    :return: A pandas DataFrame with duplicate columns removed.
    """
    # Convert threshold to a percentage for rapidfuzz
    threshold = threshold * 100

    # Create a set to keep track of columns to drop
    columns_to_drop = set()
    columns_to_rename = {}

    colnames_df = list(df.columns)

    # Iterate through columns to compare each with others
    for i, col1 in enumerate(colnames_df):

        if col1 in columns_to_drop:
            continue

        for col2 in colnames_df[i + 1 :]:

            if col2 in columns_to_drop:
                continue

            # Calculate the similarity between column contents
            str_col1 = " ".join([str(i) for i in df[col1]])
            str_col2 = " ".join([str(i) for i in df[col2]])

            similarity = fuzz.ratio(str_col1, str_col2)
            if similarity >= threshold:
                # If columns are similar, mark one for dropping
                columns_to_drop.add(col2)
                # Determine new name for the retained column
                base_name = col1.split("_")[0] if "_" in col1 else col1
                columns_to_rename[col1] = base_name

    # Remove Score and ID columns too

    # Drop the identified duplicate columns
    # print("To drop:", columns_to_drop)
    df = df.drop(columns=columns_to_drop, axis="columns")

    # Rename columns to their base names
    df = df.rename(columns=columns_to_rename)

    return df
