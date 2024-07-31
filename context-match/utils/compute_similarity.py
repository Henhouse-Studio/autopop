import sys
import pprint
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from utils.make_embeddings import *
from sentence_transformers import util
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_similarity(embedding1: np.array, embedding2: np.array):
    """
    Compute the cosine similarity between two embeddings.

    :param embedding1: The first embedding (np.array).
    :param embedding1: The second embedding (np.array).
    :return: The cosine similarity score (list).
    """

    return util.pytorch_cos_sim(embedding1, embedding2).squeeze().cpu().tolist()


def compute_similarity_matrix(embeddings1: list, embeddings2: list):
    """
    Compute the cosine similarities between two embedding sets.

    :param embedding1: The first embedding (list of np.arrays).
    :param embedding1: The second embedding (list of np.arrays).
    :return: The cosine similarity matrix (dict).
    """

    # Convert embeddings to numpy arrays if they are not already
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)

    # Compute the cosine distance between each pair of embeddings
    cosine_distances = cdist(embeddings1, embeddings2, "cosine")

    # Convert distances to similarities
    cosine_similarities = 1 - cosine_distances

    # Prepare similarity scores in a dictionary format
    similarity_scores = {
        (i, j): float(cosine_similarities[i, j])
        for i in range(cosine_similarities.shape[0])
        for j in range(cosine_similarities.shape[1])
    }

    return similarity_scores


def compute_row_embeddings(df: pd.DataFrame, column_name: str):
    """
    Compute the embeddings for each row based on a specific column.

    :param df: The dataframe to compute the embeddings for (pd.DataFrame).
    :param column_name: The name of the main column to use (str).
    :return: The embedding list for the dataframe (list).
    """

    items = df[column_name].astype(str).tolist()
    embeddings = []
    with ThreadPoolExecutor() as executor:
        for result in tqdm(
            executor.map(compute_embedding, items),
            total=len(items),
            desc=f"Computing row embeddings for '{column_name}'",
        ):
            embeddings.append(result)

    return embeddings


def compute_column_embeddings(df: pd.DataFrame, desc: str):
    """
    Compute the embeddings for each column in a specific dataframe.

    :param df: The dataframe to compute the embeddings for (pd.DataFrame).
    :param desc: The name of the database (str).
    :return sorted_columns: The keys for sorting the embeddings (list).
    :return df_embeddings: The embedding list for the dataframe (list).
    """

    future_to_col = {}
    embeddings_dict = {}

    with ThreadPoolExecutor() as executor:
        for col in df.columns:
            future = executor.submit(
                compute_embedding, ", ".join(df[col].astype(str).tolist())
            )
            future_to_col[future] = col

        for future in tqdm(
            as_completed(future_to_col),
            total=len(future_to_col),
            desc=f"Computing column embeddings from {desc} database",
        ):
            col = future_to_col[future]
            embedding = future.result()
            embeddings_dict[col] = embedding

    sorted_columns = sorted(embeddings_dict.keys())
    df_embeddings = [embeddings_dict[col] for col in sorted_columns]

    return df_embeddings, sorted_columns


def compute_similarity_softmax(df_base: pd.DataFrame, df_populate: pd.DataFrame):
    """
    Compute the softmaxed similarity scores between the rows of two dataframes.

    :param df_base: The dataframe to enrich (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment (pd.DataFrame).
    :return softmax_scores_dict: The dict containing the similarity scores for matching (dict).
    It has the following format: '(idx_df1, idx_df2): score' between all entries.
    :return highest_similar_col_name: The column name used as the merging basis.
    """

    print(
        "Computing similarity scores between the top 2 dataframes. This may take a while..."
    )
    
    df_first_embeddings, df_first_columns = compute_column_embeddings(
        df_base, "First database"
    )
    df_second_embeddings, df_second_columns = compute_column_embeddings(
        df_populate, "Second database"
    )

    # Get the similarity scores
    similarities = 1 - cdist(
        np.vstack(df_first_embeddings), np.vstack(df_second_embeddings), metric="cosine"
    )

    # save similarities to csv
    similarities_df = pd.DataFrame(similarities, columns=df_second_columns, index=df_first_columns)
    similarities_df.to_csv("similarities.csv")

    # Find the index of the maximum similarity (minimum distance)
    max_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
    highest_similar_col_name = (
        df_first_columns[max_idx[0]],
        df_second_columns[max_idx[1]],
    )

    print("highest_similar_col_name", highest_similar_col_name)

    # Compute embeddings for each row in the relevant columns of df_base and df_populate
    df_base_first_col_embeddings = compute_row_embeddings(
        df_base, highest_similar_col_name[0]
    )
    df_second_highest_col_embeddings = compute_row_embeddings(
        df_populate, highest_similar_col_name[1]
    )

    # Compute similarity scores between each embedding of df_base_first_col_embeddings and df_second_highest_col_embeddings
    similarity_scores = compute_similarity_matrix(
        df_base_first_col_embeddings, df_second_highest_col_embeddings
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
        (int(keys[i][0]), int(keys[i][1])): float(round(softmax_scores[i], 4)) for i in range(len(keys))
    }
    pprint.pprint(softmax_scores_dict)
    print("Process finished!\n")

    return softmax_scores_dict, highest_similar_col_name
