import json
from utils.filter_names import *
from utils.fetch_table_notion import *
from notion_client import Client
from argparse import ArgumentParser

# Argparser arguments:


# Execution
if __name__ == "__main__":

    # Initialize the Notion client
    with open("notion.json") as f:
        NOTION_TOKEN = json.load(f)["KEY"]

    notion = Client(auth=NOTION_TOKEN)

    # Replace with your Notion database ID
    database_id = "6ead038babe946b99854dba84ecf05a9"

    # Fetch the page links from the database
    page_names, page_links = get_page_links(notion, database_id)

    # Fetch the table links from each page
    page_table_links = get_table_links_from_pages(notion, page_links)

    # Print the links
    df_dict = {}
    for idx, tables in enumerate(page_table_links.values()):

        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"
            df = get_table_notion(NOTION_TOKEN, temp)

            df_dict[page_names[idx]] = df

    print(df_dict)
