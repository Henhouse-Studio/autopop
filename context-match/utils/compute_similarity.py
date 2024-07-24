import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from utils.make_embeddings import *
from sentence_transformers import util
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_similarity(embedding1, embedding2):

    return util.pytorch_cos_sim(embedding1, embedding2).squeeze().cpu().tolist()


def compute_similarity_matrix(embeddings1, embeddings2):

    similarity_scores = {}
    for idx1, emb1 in enumerate(embeddings1):
        for idx2, emb2 in enumerate(embeddings2):
            similarity_scores[(idx1, idx2)] = (
                util.pytorch_cos_sim(emb1, emb2).squeeze().cpu().tolist()
            )

    return similarity_scores


def get_embeddings(df, column):

    items = df[column].astype(str).tolist()
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(compute_embedding, items))

    return embeddings


def compute_similarity_softmax(df_base: pd.DataFrame, df_populate: pd.DataFrame):

    print(
        "Computing similarity scores between the top 2 dataframes. This may take a while..."
    )

    # Get the last column of df_base and join its items into a string
    col_name_df1 = df_base.columns[0]
    str_last_col_df1 = ", ".join(df_base[col_name_df1].astype(str).tolist())

    # Compute embedding for the concatenated string of the last column of df_base
    str_last_col_df1_embedding = compute_embedding(str_last_col_df1)

    # Compute embeddings for each column in df_populate in parallel
    df_second_embeddings = {}
    with ThreadPoolExecutor() as executor:

        future_to_col = {
            executor.submit(
                compute_embedding, ", ".join(df_populate[col].astype(str).tolist())
            ): col
            for col in df_populate.columns
        }
        for future in tqdm(
            as_completed(future_to_col),
            total=len(future_to_col),
            desc="Computing column embeddings",
        ):
            col = future_to_col[future]
            df_second_embeddings[col] = future.result()

    # Compute similarity scores between str_last_col_df1_embedding and df_second_embeddings
    similarity_scores = {
        col: util.pytorch_cos_sim(str_last_col_df1_embedding, emb)
        .squeeze()
        .cpu()
        .tolist()
        for col, emb in df_second_embeddings.items()
    }

    # Sort the similarity scores and get the highest similarity score column name
    highest_similar_col_name = max(similarity_scores, key=similarity_scores.get)

    # Compute embeddings for each row in the relevant columns of df_base and df_populate
    df_first_last_col_embeddings = get_embeddings(df_base, col_name_df1)
    df_second_highest_col_embeddings = get_embeddings(
        df_populate, highest_similar_col_name
    )

    # Compute similarity scores between each embedding of df_first_last_col_embeddings and df_second_highest_col_embeddings
    similarity_scores = compute_similarity_matrix(
        df_first_last_col_embeddings, df_second_highest_col_embeddings
    )

    # Sort the similarity scores per each entry in descending order and then by the first element of the key in increasing order
    similarity_scores = dict(
        sorted(similarity_scores.items(), key=lambda x: (x[0][0], -x[1]))
    )

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
    softmax_scores_dict = {
        tuple(keys[i]): round(softmax_scores[i], 4) for i in range(len(keys))
    }

    print("Process finished!\n")

    return softmax_scores_dict, highest_similar_col_name
