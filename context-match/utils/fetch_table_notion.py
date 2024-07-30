import os
import notion_df
import numpy as np
import pandas as pd
from utils.make_embeddings import *
from utils.compute_similarity import *


def get_table_notion(NOTION_TOKEN, DATABASE_URL):

    df = notion_df.download(DATABASE_URL, api_key=NOTION_TOKEN)
    df = df.iloc[:, ::-1]

    return df


def get_page_links(notion_client, database_id):
    """
    Get links to pages in a Notion database.

    :param database_id: The ID of the Notion database.
    :return: A list of links to the pages.
    """
    response = notion_client.databases.query(database_id=database_id)
    pages = response.get("results")

    names = []
    links = []
    for page in pages:

        # Extract the page information
        # print(page)
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


def get_dataframes(
    page_links: dict, page_names: list, notion_token: str
):

    print("Retrieving databases...")
    df_dict = {}
    for idx, tables in enumerate(page_links.values()):

        is_fact = page_names[idx][0]
        table_name = page_names[idx][1]
        path_table = f"databases/{table_name}.csv"

        if is_fact:
            print(f"Found Fact table '{table_name}'")
        else:
            print(f"Found table '{table_name}'")

        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"

            # when is a 'FACT' table try to load it, else download it
            if is_fact:

                # loading the table if it exists
                if os.path.isfile(path_table):
                    df = pd.read_csv(path_table)

                # downloading the table if it does not exist
                else:
                    print('Saving Fact table. This may take a while as it is a large database...')
                    df = get_table_notion(notion_token, temp)
                    df.to_csv(path_table, index=False)
            
            # when is not a 'FACT' table then just download it
            else:
                df = get_table_notion(notion_token, temp)

            # Adding to the data dictionary
            df_dict[table_name] = df

    print(f"Number of databases found: {len(df_dict)}\n")
    return df_dict


def score_dataframes(
    dfs_dic: pd.DataFrame, prompt_embedding: np.array
):  
    print("Scoring databases based on prompt...")

    df_dict = {}
    for table_name, df in dfs_dic.items():

        # Converting the table title and column names into context
        col_names = list(df.columns)
        sample = df.sample(n=1, random_state=42)
        desc = f"The name of the table is {table_name}. It has these columns and entry samples:\n"

        for col_name in col_names:
            desc += f"{col_name}: {sample[col_name].values[0]}\n"

        # print(desc)
        # Computing the embeddings and similarity scores
        field_embeddings = compute_embedding(desc)
        similarity_score = compute_similarity(prompt_embedding, field_embeddings)
        similarity_score = round(similarity_score * 100, 2)
        # print(similarity_score, table_name)
        # Adding to the data dictionary
        df_dict[table_name] = (similarity_score, df)

    # Sort the dictionary based on similarity score
    df_dict = dict(sorted(df_dict.items(), key=lambda x: x[1][0], reverse=True))

    # printing similarity score, name
    for key, value in df_dict.items():
        print(value[0], key)

    return df_dict, get_top_k(df_dict)

def get_top_k(dfs_dict_ranked):
    '''
    Get the top-k similar dataframes based on the similarity scores
    '''

    # compute the std-dev from the similarity scores
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
