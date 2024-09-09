import os
import pickle
import argparse
import pandas as pd
from utils.constants import *
from notion_client import Client
from utils.notion_df import get_table_notion 
from utils.prompt_to_openai import rerank_dataframes
from utils.progress_history import save_progress_text
def get_page_links(notion_client, database_id: str):
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

    # Get the page link via the Notion metadata
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


def get_table_links_from_pages(notion, page_links: list):
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
    page_names_file: str = "page_names.pkl",
    page_table_links_file: str = "page_table_links.pkl",
    verbose: bool = False,
):
    with open(page_names_file, "wb") as f:
        pickle.dump(page_names, f)

    with open(page_table_links_file, "wb") as f:
        pickle.dump(page_table_links, f)

    save_progress_text(
        f"Data saved to {page_names_file} and {page_table_links_file}", verbose=verbose
    )


def load_data_pickle(
    page_names_file: str = "page_names.pkl",
    page_table_links_file: str = "page_table_links.pkl",
    verbose: bool = False,
):
    with open(page_names_file, "rb") as f:
        page_names = pickle.load(f)

    with open(page_table_links_file, "rb") as f:
        page_table_links = pickle.load(f)

    save_progress_text(
        f"Data loaded from {page_names_file} and {page_table_links_file}",
        verbose=verbose,
    )

    return page_names, page_table_links


def get_dataframes(
    notion_token: str,
    database_id: str,
    args: argparse.Namespace,
    page_names_file: str = os.path.join(DATA_DIR, "page_names.pkl"),
    page_table_links_file: str = os.path.join(DATA_DIR, "page_table_links.pkl"),
    verbose: bool = False,
):
    """
    Retrieve dataframes from Notion pages and store them locally if not already saved.

    This function connects to a Notion database using the provided API token and database ID,
    retrieves the relevant pages and table links, and processes them into pandas DataFrames.
    If the data has been previously saved locally, it loads the data from pickle files to avoid
    redundant API calls.

    The function performs the following steps:
    1. Checks if saved data files exist and loads them if available, unless `fetch_tables`
       is set to True in `args`.
    2. If saved data is not available, it initializes a Notion client, retrieves page names and
       table links, and saves this data for future use.
    3. For each table link, it checks if a corresponding CSV file exists locally:
       - If the file exists and `fetch_tables` is False, it loads the DataFrame from the CSV file.
       - If the file does not exist or `fetch_tables` is True, it fetches the table data from Notion
         and saves it as a CSV file.
    4. It organizes the tables into a dictionary with table names as keys and tuples containing
       a boolean indicating if it is a "Fact" table and the corresponding DataFrame as values.

    :param notion_token: The Notion API token.
    :param database_id: The ID of the Notion database.
    :param args: Argparse namespace containing parameters, including 'fetch_tables'.
    :param page_names_file: Path to the pickle file for storing page names (default: "page_names.pkl").
    :param page_table_links_file: Path to the pickle file for storing page-table links (default: "page_table_links.pkl").
    :param verbose: If True, prints detailed logs during execution (default: False).

    :return: A dictionary with table names as keys and tuples of (is_fact, DataFrame) as values.
    """
    # Indicator message
    save_progress_text("Retrieving databases...\n", verbose=verbose)

    if (
        os.path.exists(page_names_file)
        and os.path.exists(page_table_links_file)
        and not args.fetch_tables
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
        path_table = os.path.join(DATA_DIR, f"{table_name}.csv")
        save_progress_text(
            f"[{idx+1}] Found{' Fact' * is_fact} table: '{table_name}'", verbose=verbose
        )

        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"

            if is_fact:

                if os.path.isfile(path_table) and not args.fetch_tables:

                    df = pd.read_csv(path_table)

                else:
                    save_progress_text(
                        "Saving Fact table. This may take a while as it is a large database...",
                        verbose=verbose,
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

    save_progress_text(
        f"[*] Number of databases found: {len(df_dict)}\n", verbose=verbose
    )

    return df_dict


def main_sort_dataframes(
    dfs_dict: dict, enriched_prompt: str, openai_token: str, verbose: bool = False
):
    """
    Sort each dataframe based on how relevant they are to the prompt.

    :param dfs_dict: A dictionary with table names as keys and pandas DataFrames as values (dict).
    :param enriched_prompt: The enriched prompt text (str).
    :param openai_token: The token to access ChatGPT (str).
    :param verbose: If True, prints detailed logs during execution (default: False).
    :return: A tuple containing the sorted and fact dataframes (dict, dict).
    """

    save_progress_text("Selecting most relevant databases...", verbose=verbose)

    df_dict_new = {}
    df_fact_dict = {}

    # Generate a description for all databases
    for table_name, (is_fact, df) in dfs_dict.items():

        col_names = list(df.columns)
        sample = df.sample(n=2, random_state=42)
        desc = f"The name of the table is {table_name}. This is a extract from the table:\n"

        for sample_value in range(len(sample)):
            for col_name in col_names:
                desc += f"{col_name}: {sample[col_name].values[sample_value]}\n"

        if is_fact:
            df_fact_dict[table_name] = (df, desc)

        df_dict_new[table_name] = (df, desc)

    # Names of the relevant tables based on prompt
    relevant_tables = rerank_dataframes(enriched_prompt, df_dict_new, openai_token)
    relevant_fact_tables = rerank_dataframes(
        enriched_prompt, df_fact_dict, openai_token, is_fact=True
    )

    df_ranked = {table_name: df_dict_new[table_name] for table_name in relevant_tables}
    df_fact_ranked = {
        table_name: df_fact_dict[table_name]
        for table_name in relevant_fact_tables
        if table_name not in relevant_tables
    }

    # Print if verbose
    top_text = f"Selecting Top-{len(relevant_tables)} tables:\n"

    for i, (table_name, _) in enumerate(df_ranked.items()):
        top_text += f"[{i+1}]: {table_name}\n"

    save_progress_text(top_text, verbose=verbose)

    fact_text = f"\nSelecting Top-{len(relevant_fact_tables)} Fact tables:\n"

    for i, (table_name, _) in enumerate(df_fact_ranked.items()):
        fact_text += f"[{i+1}]: {table_name}\n"

    save_progress_text(fact_text, verbose=verbose)

    return df_ranked, df_fact_ranked
