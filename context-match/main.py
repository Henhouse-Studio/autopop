import json
from notion_client import Client
from argparse import ArgumentParser
from utils.filter_names import *
from utils.make_embeddings import *
from utils.prompt_expansion import *
from utils.fetch_table_notion import *
from sklearn.metrics.pairwise import cosine_similarity

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

    # Prompt from the user
    prompt = "Get me a table of people's job profiles and their blogposts"

    # Enrichment of the prompt
    prompt = expand_prompt(prompt)
    print(prompt)
    prompt_embedding = compute_embedding(prompt)

    # Converting the databases to pandas dataframes
    df_dict = {}
    for idx, tables in enumerate(page_table_links.values()):

        print(f"Found table '{page_names[idx]}'")
        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"
            df = get_table_notion(NOTION_TOKEN, temp)

            # Converting the table title and column names into context
            colnames = list(df.columns)
            sample = df.sample(n=1)
            desc = f"The name of the table is {page_names[idx]}. It has these columns and entry samples:\n"

            for colname in colnames:

                desc += f"{colname}: {sample[colname].values[0]}\n"

            # print(desc)
            # Computing the embeddings and similarity scores
            field_embeddings = compute_embedding(desc)
            similarity_score = cosine_similarity(
                [prompt_embedding], [field_embeddings]
            )[0][0]
            similarity_score = round(similarity_score * 100, 2)
            # print(similarity_score)
            # Adding to the data dictionary
            df_dict[page_names[idx]] = (similarity_score, df)

    print(f"Number of databases found: {len(df_dict)}")

    # Get top 5 tables
