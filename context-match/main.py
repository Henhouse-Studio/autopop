import json
from utils.filter_names import *
from utils.fetch_table_notion import *
from utils.make_embeddings import *
from notion_client import Client
from argparse import ArgumentParser

# Argparser arguments:


# Execution
if __name__ == "__main__":
    
    # Initialize the Notion client
    with open("notion.json") as f:
        NOTION_TOKEN = json.load(f)["KEY"]

    notion = Client(auth=NOTION_TOKEN)

    # The Notion database ID
    database_id = "6ead038babe946b99854dba84ecf05a9"

    # Get the page and table links from the database
    page_names, page_links = get_page_links(notion, database_id)
    page_table_links = get_table_links_from_pages(notion, page_links)

    # Prompt user inputs:
    prompt = "Get me a table of people, where they work and their blogs"

    # Converting the databases to pandas dataframes
    df_dict = {}
    for idx, tables in enumerate(page_table_links.values()):

        print(f"Found table '{page_names[idx]}'")
        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"
            df = get_table_notion(NOTION_TOKEN, temp)

            context = [page_names[idx]] + list(df.columns)
            prompt_embedding, field_embeddings = compute_embeddings(context, prompt)

            df_dict[page_names[idx]] = df

            

    print(f"Number of databases found: {len(df_dict)}")





