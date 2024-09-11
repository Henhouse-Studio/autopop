import pandas as pd
from uuid import UUID
from notion_client import Client
from urllib.parse import urlparse

def get_id(url: str) -> str:
    """Return the id of the object behind the given URL."""
    parsed = urlparse(url)
    if parsed.netloc not in ("notion.so", "www.notion.so"):
        raise ValueError("Not a valid Notion URL.")
    path = parsed.path
    if len(path) < 32:
        raise ValueError("The path in the URL seems to be incorrect.")
    raw_id = path[-32:]
    return str(UUID(raw_id))

def extract_content(prop):
    if prop['type'] == 'rich_text':
        return prop['rich_text'][0]['text']['content'] if prop['rich_text'] else ''
    if prop['type'] == 'multi_select':
        return ", ".join([option['name'] for option in prop['multi_select']])
    if prop['type'] == 'select':
        return prop['select']['name'] if prop['select'] else ''
    if prop['type'] == 'title':
        return prop['title'][0]['text']['content'] if prop['title'] else ''
    if prop['type'] == 'date':
        start_date = prop['date']['start'] if prop['date'] and 'start' in prop['date'] else None
        end_date = prop['date']['end'] if prop['date'] and 'end' in prop['date'] else None
        if start_date and end_date:
            return f"{start_date} to {end_date}"
        elif start_date:
            return start_date
        return None
    if prop['type'] == 'number':
        return str(prop['number'])
    if prop['type'] == 'formula':
        return prop['formula']['string'] if prop['formula'] and 'string' in prop['formula'] else str(prop['formula'].get('number', ''))
    if prop['type'] == 'relation':
        return ", ".join([relation['id'] for relation in prop['relation']])
    if prop['type'] == 'people':
        return ", ".join([person['name'] for person in prop['people']])
    if prop['type'] == 'files':
        return ", ".join([file['name'] for file in prop['files']])
    if prop['type'] == 'checkbox':
        return 'Yes' if prop['checkbox'] else 'No'
    if prop['type'] == 'url':
        return prop['url']
    if prop['type'] == 'email':
        return prop['email']
    if prop['type'] == 'phone_number':
        return prop['phone_number']
    if prop['type'] == 'created_time':
        return prop['created_time']
    if prop['type'] == 'last_edited_time':
        return prop['last_edited_time']
    if prop['type'] == 'created_by':
        return prop['created_by']['name'] if prop['created_by'] else ''
    if prop['type'] == 'last_edited_by':
        return prop['last_edited_by']['name'] if prop['last_edited_by'] else ''
    if prop['type'] == 'rollup':
        # Rollup can return multiple types, handle it based on the `rollup` type
        if 'array' in prop['rollup']:
            return ", ".join([extract_content(item) for item in prop['rollup']['array']])
        if 'number' in prop['rollup']:
            return prop['rollup']['number']
        if 'date' in prop['rollup']:
            return prop['rollup']['date']['start'] if 'start' in prop['rollup']['date'] else None
    return None  # Default for unsupported types


def get_table_notion(NOTION_TOKEN: str, DATABASE_URL: str) -> pd.DataFrame:
    """
    Retrieve a table from Notion as a DataFrame.

    :param NOTION_TOKEN: The Notion API token.
    :param DATABASE_URL: The URL of the Notion database.
    :return: A pandas DataFrame containing the data from the Notion database.
    """

    notion = Client(auth=NOTION_TOKEN)
    database_id = get_id(DATABASE_URL)
    notion_data = notion.databases.query(database_id=database_id)['results']

    # Convert the Notion data into a DataFrame
    rows = [{key: extract_content(prop) for key, prop in page['properties'].items()} for page in notion_data]

    # Create DataFrame and reorder columns based on the first record's order
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df.columns[::-1]]  # Reverse column order
    return df