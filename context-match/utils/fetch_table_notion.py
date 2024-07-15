import json
import notion_df
from notion_client import Client


def get_table_notion(NOTION_TOKEN, DATABASE_URL):

    return notion_df.download(DATABASE_URL, api_key=NOTION_TOKEN)


def get_page_links(database_id):
    """
    Get links to pages in a Notion database.

    :param database_id: The ID of the Notion database.
    :return: A list of links to the pages.
    """
    response = notion.databases.query(database_id=database_id)
    pages = response.get("results")

    links = []
    for page in pages:

        page_id = page["id"]
        # Extract the page URL
        page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
        links.append(page_url)

    return links


def get_table_links_from_pages(page_links):
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


if __name__ == "__main__":

    # Initialize the Notion client
    with open("notion.json") as f:
        NOTION_TOKEN = json.load(f)["KEY"]

    notion = Client(auth=NOTION_TOKEN)

    # Replace with your Notion database ID
    database_id = "6ead038babe946b99854dba84ecf05a9"

    # Fetch the page links from the database
    page_links = get_page_links(database_id)

    # Fetch the table links from each page
    page_table_links = get_table_links_from_pages(page_links)

    # Print the links
    for page, tables in page_table_links.items():

        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"
            df = get_table_notion(NOTION_TOKEN, temp)

            print(list(df.columns)[-1])
