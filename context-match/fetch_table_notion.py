from notion_client import Client

from basic_notion.query import Query
from basic_notion.page import NotionPage, NotionPageList
from basic_notion.field import SelectField, TitleField, NumberField

# First define models

class MyRow(NotionPage):
    name = TitleField(property_name='Name')
    subscription = SelectField(property_name='Subscription')
    some_number = NumberField(property_name='Some Number')
    # ... your other fields go here
    # See your database's schema and the available field classes
    # in basic_notion.field to define this correctly.

class MyData(NotionPageList[MyRow]):
    ITEM_CLS = MyRow

# You need to create an integration and get an API token from Notion:
NOTION_TOKEN = '<your-notion-api-token>'
DATABASE_ID = '<your-database-ID>'

# Now you can fetch the data

def get_data(database_id: str) -> MyData:
    client = Client(auth=NOTION_TOKEN)
    data = client.databases.query(
        **Query(database_id=database_id).filter(
            # Some filter here
            MyRow.name.filter.starts_with('John')
        ).sorts(
            # You can sort it here
            MyRow.name.sort.ascending
        ).serialize()
    )
    return MyData(data=data)


my_data = get_data()
for row in my_data.items():
    print(f'{row.name.get_text()} - {row.some_number.number}')
# Do whatever else you may need to do