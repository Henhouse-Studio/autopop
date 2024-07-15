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
        # Normally, you would need to fetch page content and parse it to find table blocks
        # Notion API does not provide direct way to get table links, this is a workaround
        page_id = page_link.split("/")[-1]
        blocks = notion.blocks.children.list(block_id=page_id).get("results")

        table_links = []
        for block in blocks:
            if block["type"] == "table":
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

    for link in page_links:

        df = get_table_notion(NOTION_TOKEN, link)
        print(df.head())

    # # Fetch the table links from each page
    # page_table_links = get_table_links_from_pages(page_links)

    # # Print the links
    # for page, tables in page_table_links.items():
    #     print(f"URL: {page}")
    #     for table in tables:
    #         print(f"  Table: {table}")
