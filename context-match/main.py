import json
from notion_client import Client
from argparse import ArgumentParser
from utils.filter_names import *
from utils.make_embeddings import *
from utils.prompt_expansion import *
from utils.fetch_table_notion import *
from sentence_transformers import util

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
    prompt = "Get me a table of firms and their employees"
    # prompt = "Get me a table of people's job profiles"

    # Enrichment of the prompt
    prompt = expand_prompt_with_synonyms(prompt)
    # print(prompt)
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
            sample = df.sample(n=1, random_state=42)
            desc = f"The name of the table is {page_names[idx]}. It has these columns and entry samples:\n"

            for colname in colnames:

                desc += f"{colname}: {sample[colname].values[0]}\n"

            # print(desc)
            # Computing the embeddings and similarity scores
            field_embeddings = compute_embedding(desc)
            similarity_score = util.pytorch_cos_sim(prompt_embedding, field_embeddings).squeeze().cpu().tolist()
            similarity_score = round(similarity_score * 100, 2)
            # print(similarity_score)
            # Adding to the data dictionary
            df_dict[page_names[idx]] = (similarity_score, df)
            
    # Sort the dictionary based on similarity score
    df_dict = dict(sorted(df_dict.items(), key=lambda x: x[1][0], reverse=True))

    print(f"Number of databases found: {len(df_dict)}")

    # get the first item of the df_dict
    df_ranked = list(df_dict.items())
    df_first = df_ranked[0][1][1]
    df_second = df_ranked[1][1][1]

    # get the last column of the df_raw
    field_last_col_df1 = df_first.columns[-1]
    
    # join the the column items from df_first into a string
    str_last_col_df1 = ', '.join(df_first[field_last_col_df1].astype(str).tolist())

    # mk embeddings of str_last_col_df1
    str_last_col_df1_embedding = compute_embedding(str_last_col_df1)

    # mk embeddings per column of df_second
    df_second_embeddings = {}
    for col in df_second.columns:
        df_second_embeddings[col] = compute_embedding(', '.join(df_second[col].astype(str).tolist()))

    # compute similarity scores between str_last_col_df1_embedding and df_second_embeddings
    similarity_scores = {}
    for col, emb in df_second_embeddings.items():
        similarity_scores[col] = util.pytorch_cos_sim(str_last_col_df1_embedding, emb).squeeze().cpu().tolist()

    # sort the similarity scores
    similarity_scores = dict(sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True))

    # get the highest similarity score name
    highest_similar_col_name = list(similarity_scores.items())[0][0]

    


