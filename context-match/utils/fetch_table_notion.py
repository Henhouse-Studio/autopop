import notion_df


def get_table_notion(NOTION_TOKEN, DATABASE_URL):
    return notion_df.download(DATABASE_URL, api_key=NOTION_TOKEN)
