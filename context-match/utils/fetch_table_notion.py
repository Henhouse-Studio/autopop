import notion_df


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
    tags = []
    for page in pages:

        # Extract the page information
        # print(page)
        page_title = page["properties"]["Name"]["title"][0]["text"]["content"]
        page_url = f"https://www.notion.so/{page["id"].replace('-', '')}"
        page_tags = page["properties"]["Tags"]["multi_select"]
        tag_list = []
        for tag in page_tags:

            tag_list.append(tag["name"])

        names.append(page_title)
        links.append(page_url)
        tags.append(tag_list)

    return names, links, tags


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
