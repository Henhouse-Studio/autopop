import json
from utils.filter_names import *
from utils.fetch_table_notion import *
from utils.make_embeddings import *
from utils.compute_similarity import *
from notion_client import Client
from argparse import ArgumentParser
import time
import numpy as np

# Argparser arguments:


# Execution
if __name__ == "__main__":

    start_time = time.time()
    
    # Initialize the Notion client
    with open("notion.json") as f:
        NOTION_TOKEN = json.load(f)["KEY"]

    notion = Client(auth=NOTION_TOKEN)

    print(f"Initialization time: {time.time() - start_time:.2f} seconds")
    init_time = time.time()

    # The Notion database ID
    database_id = "6ead038babe946b99854dba84ecf05a9"

    # Get the page and table links from the database
    page_names, page_links = get_page_links(notion, database_id)
    page_table_links = get_table_links_from_pages(notion, page_links)

    print(f"Fetching page and table links time: {time.time() - init_time:.2f} seconds")
    fetch_links_time = time.time()

    # Prompt user inputs:
    prompt = "Get me a table of people, where they work and their blogs"
    prompt_embedding = compute_embedding(prompt)
    print(prompt_embedding.shape)

    # Save the prompt embedding in npy file
    np.save("prompt_embedding.npy", prompt_embedding)
    
    # load prompt embedding if already saved
    prompt_embedding = np.load("prompt_embedding.npy")    

    print(f"Computing prompt embedding time: {time.time() - fetch_links_time:.2f} seconds")

    # Converting the databases to pandas dataframes
    df_dict = {}
    for idx, tables in enumerate(page_table_links.values()):

        print(f"Found table '{page_names[idx]}'")
        for table in tables:

            id = table.split("#")
            temp = f"https://www.notion.so/about-/{id[-1]}?v=eceee883ed684a75831aec55806e39d2"
            df = get_table_notion(NOTION_TOKEN, temp)

            context = [page_names[idx]] + list(df.columns)
            # Join the context to form a single string
            context = ", ".join(context)
            
            field_embeddings = compute_embedding(context)
            # similarity_score = compute_similarity(prompt_embedding, field_embeddings)

            print(field_embeddings.shape)

            # df_dict[page_names[idx]] = (similarity_score, df)

            

    print(f"Number of databases found: {len(df_dict)}")

    # Get top 5 tables





