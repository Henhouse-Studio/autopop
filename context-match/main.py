import json
import numpy as np
from scipy.special import softmax
from notion_client import Client
from argparse import ArgumentParser
from utils.filter_names import *
from utils.make_embeddings import *
from utils.prompt_expansion import *
from utils.fetch_table_notion import *
from utils.compute_similarity import *
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            similarity_score = compute_similarity(prompt_embedding, field_embeddings)
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

     # Get the last column of df_first and join its items into a string
    last_col_name_df1 = df_first.columns[-1]
    str_last_col_df1 = ', '.join(df_first[last_col_name_df1].astype(str).tolist())

    # Compute embedding for the concatenated string of the last column of df_first
    str_last_col_df1_embedding = compute_embedding(str_last_col_df1)

    # Compute embeddings for each column in df_second in parallel
    df_second_embeddings = {}
    with ThreadPoolExecutor() as executor:
        future_to_col = {executor.submit(compute_embedding, ', '.join(df_second[col].astype(str).tolist())): col for col in df_second.columns}
        for future in as_completed(future_to_col):
            col = future_to_col[future]
            df_second_embeddings[col] = future.result()

    # Compute similarity scores between str_last_col_df1_embedding and df_second_embeddings
    similarity_scores = {col: util.pytorch_cos_sim(str_last_col_df1_embedding, emb).squeeze().cpu().tolist() for col, emb in df_second_embeddings.items()}

    # Sort the similarity scores and get the highest similarity score column name
    highest_similar_col_name = max(similarity_scores, key=similarity_scores.get)
    
    def get_embeddings(df, column):
        items = df[column].astype(str).tolist()
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(compute_embedding, items))
        return embeddings

    # Compute embeddings for each row in the relevant columns of df_first and df_second
    df_first_last_col_embeddings = get_embeddings(df_first, last_col_name_df1)
    df_second_highest_col_embeddings = get_embeddings(df_second, highest_similar_col_name)

    # Compute similarity scores between each embedding of df_first_last_col_embeddings and df_second_highest_col_embeddings
    similarity_scores = compute_similarity_matrix(df_first_last_col_embeddings, df_second_highest_col_embeddings)


    # Sort the similarity scores per each entry in descending order and then by the first element of the key in increasing order
    similarity_scores = dict(sorted(similarity_scores.items(), key=lambda x: (x[0][0], -x[1])))

    # Convert the dictionary to a structured array for vectorized operations
    keys, scores = zip(*similarity_scores.items())
    keys = np.array(keys)
    scores = np.array(scores)

    # Compute softmax for each unique key[0]
    unique_keys_a = np.unique(keys[:, 0])
    softmax_scores = np.zeros_like(scores)

    # Compute softmax scores in a vectorized manner
    for key_a in unique_keys_a:
        mask = keys[:, 0] == key_a
        softmax_scores[mask] = softmax(scores[mask])

    # Reconstruct the dictionary with rounded softmax scores
    softmax_scores_dict = {tuple(keys[i]): round(softmax_scores[i], 4) for i in range(len(keys))}

    # Threshold based on table size
    threshold = 2 * 0.8 / len(df_second[highest_similar_col_name])

    # Filter similarity scores based on threshold
    filtered_similarity_scores = {k: v for k, v in softmax_scores_dict.items() if v >= threshold}

    print(filtered_similarity_scores)
    print(len(filtered_similarity_scores))

    


