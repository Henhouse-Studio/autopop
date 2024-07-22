import notion_df
import numpy as np
import pandas as pd
from utils.make_embeddings import *
from utils.compute_similarity import *


def get_table_notion(NOTION_TOKEN, DATABASE_URL):

    return notion_df.download(DATABASE_URL, api_key=NOTION_TOKEN)


def get_page_links(notion, database_id):
    """
    Get links to pages in a Notion database.

    :param database_id: The ID of the Notion database.
    :return: A list of links to the pages.
    """
    response = notion.databases.query(database_id=database_id)
    pages = response.get("results")

    names = []
    links = []
    for page in pages:

        # Extract the page information
        # print(page)
        page_title = page["properties"]["Name"]["title"][0]["text"]["content"]
        page_id = page["id"]
        page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
        names.append(page_title)
        links.append(page_url)

    return names, links


def get_table_links_from_pages(notion, page_links):
    """
    Get links to tables in each Notion page.

    :param page_links: A list of links to the pages.
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


def to_pandas(
    page_links: dict, page_names: list, prompt_embedding: np.array, notion_token: str
):

    print("Retrieving databases...")
    df_dict = {}
    for idx, tables in enumerate(page_links.values()):

        print(f"Found table '{page_names[idx]}'")
        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"
            df = get_table_notion(notion_token, temp)

            # Converting the table title and column names into context
            colnames = list(df.columns)
            sample = df.sample(n=1, random_state=42)
            desc = f"The name of the table is {page_names[idx]}. It has these columns and entry samples:\n"

            for colname in colnames:

                desc += f"{colname}: {sample[colname].values[0]}\n"

            # print(desc)
            # Computing the embeddings and similarity scores
            field_embeddings = compute_embedding(desc)
            similarity_score = compute_similarity(prompt_embedding, field_embeddings)
            similarity_score = round(similarity_score * 100, 2)
            # print(similarity_score)
            # Adding to the data dictionary
            df_dict[page_names[idx]] = (similarity_score, df)

    # Sort the dictionary based on similarity score
    df_dict = dict(sorted(df_dict.items(), key=lambda x: x[1][0], reverse=True))
    print(f"Number of databases found: {len(df_dict)}\n")

    return df_dict


def remove_duplicates(df: pd.DataFrame, threshold: float = 0.9):

    colnames_df = list(df.columns)
    colnames_pop = [colname for colname in colnames_df if "_df2" in colname]
    colnames_base = list(set(colnames_df) - set(colnames_pop))

    for colname_b in colnames_base:

        colnames_pop_copy = colnames_pop.copy()
        for colname_p in colnames_pop_copy:
            # Calculate similarity between columns
            matches = df[colname_b] == df[colname_p]
            similarity_ratio = matches.sum() / len(matches)

            if similarity_ratio >= threshold:
                df.drop(colname_p, axis="columns", inplace=True)
                # Update colnames_pop after dropping the column
                colnames_pop.remove(colname_p)

    # Rename columns to remove '_df2'
    df.rename(
        columns=lambda x: (
            x.replace("_df2", "") if x.replace("_df2", "") not in colnames_base else x
        ),
        inplace=True,
    )

    return df
