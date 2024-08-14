import sys
import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
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


def compute_embeddings_rows(df: pd.DataFrame, desc: str = "a"):
    """
    Compute the embeddings based on a specific column.

    :param df: The dataframe to compute the embeddings for (pd.DataFrame).
    :param desc: The name of the database (str).
    :return: The embedding list for the dataframe rows (list).
    """

    future_to_row = {}
    embeddings_dict = {}

    with ThreadPoolExecutor() as executor:

        for idx, row in df.iterrows():
            future = executor.submit(
                compute_embedding, ", ".join(row.astype(str).tolist())
            )
            future_to_row[future] = idx

        for future in tqdm(
            as_completed(future_to_row),
            total=len(future_to_row),
            desc=f"Computing row embeddings from {desc} database",
        ):
            idx = future_to_row[future]
            embedding = future.result()
            embeddings_dict[idx] = embedding

    sorted_rows = sorted(embeddings_dict.keys())
    df_embeddings = [embeddings_dict[row] for row in sorted_rows]

    return df_embeddings


def compute_column_embeddings(df: pd.DataFrame, desc: str = "a", colname: str = ""):
    """
    Compute the embeddings for each column in a specific dataframe.

    :param df: The dataframe to compute the embeddings for (pd.DataFrame).
    :param desc: The name of the database (str).
    :param colname: The name of the specific column to calculate the embeddings for if desired (str).
    :return sorted_columns: The keys for sorting the embeddings (list).
    :return df_embeddings: The embedding list for the dataframe based on the columns (list).
    """

    future_to_col = {}
    embeddings_dict = {}

    with ThreadPoolExecutor() as executor:

        columns = (
            colname if ((colname != "") and (colname in df.columns)) else df.columns
        )

        for col in columns:
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


def compute_similarity_entries_row(
    df_base: pd.DataFrame, df_populate: pd.DataFrame, verbose: bool = False
):
    """
    Compute the similarity scores between the rows of two dataframes.

    :param df_base: The dataframe to enrich (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment (pd.DataFrame).
    :param verbose: Whether to print the scores (bool, default = False).
    :return softmax_scores_dict: The dict containing the similarity scores for matching (dict).
    It has the following format: '(idx_df1, idx_df2): score' between all entries.
    :return highest_similar_col_name: The column name used as the merging basis.
    """

    # Compute embeddings for each row in the relevant columns of df_base and df_populate
    df_base_row_embeddings = compute_embeddings_rows(df_base)
    df_pop_row_embeddings = compute_embeddings_rows(df_populate)

    # Compute similarity scores between each embedding of df_base_first_col_embeddings and df_pop_highest_col_embeddings
    similarity_scores = compute_similarity_matrix(
        df_base_row_embeddings, df_pop_row_embeddings
    )

    # Sort the similarity scores per each entry in descending order and then by the first element of the key in increasing order
    similarity_scores = dict(
        sorted(similarity_scores.items(), key=lambda x: (x[0][0], -x[1]))
    )

    converted_scores = {
        (int(key[0]), int(key[1])): float(value)
        for key, value in similarity_scores.items()
    }

    if verbose:
        pprint.pprint(converted_scores)

    print("Finished computing row similarities!\n")

    return converted_scores


# Unused
def compute_similarity_entries_col(df_base: pd.DataFrame, df_populate: pd.DataFrame):
    """
    Compute the similarity scores between the rows of two dataframes.

    :param df_base: The dataframe to enrich (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment (pd.DataFrame).
    :return softmax_scores_dict: The dict containing the similarity scores for matching (dict).
    It has the following format: '(idx_df1, idx_df2): score' between all entries.
    :return highest_similar_col_name: The column name used as the merging basis.
    """

    sim_matrix, sorted_col1, sorted_col2 = compute_similar_columns(df_base, df_populate)

    # Find the index of the maximum similarity (minimum distance)
    max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    most_similar_cols = (
        sorted_col1[max_idx[0]],
        sorted_col2[max_idx[1]],
    )

    print("Columns which are the most similar:", most_similar_cols)

    # Compute embeddings for each row in the relevant columns of df_base and df_populate
    df_base_first_col_embeddings, _ = compute_column_embeddings(
        df_base, sorted_col1[max_idx[0]]
    )
    df_pop_highest_col_embeddings, _ = compute_column_embeddings(
        df_populate, sorted_col2[max_idx[1]]
    )

    # Compute similarity scores between each embedding of df_base_first_col_embeddings and df_pop_highest_col_embeddings
    similarity_scores = compute_similarity_matrix(
        df_base_first_col_embeddings, df_pop_highest_col_embeddings
    )

    # Sort the similarity scores per each entry in descending order and then by the first element of the key in increasing order
    similarity_scores = dict(
        sorted(similarity_scores.items(), key=lambda x: (x[0][0], -x[1]))
    )

    converted_scores = {
        (int(key[0]), int(key[1])): float(value)
        for key, value in similarity_scores.items()
    }

    pprint.pprint(converted_scores)

    print("Finished computing row similarities!\n")

    return converted_scores, most_similar_cols


# Unused
def compute_similar_columns(df1: pd.DataFrame, df2: pd.DataFrame):

    print(
        "Computing similarity scores between dataframes 'A' and 'B'. This may take a while..."
    )

    df_first_embeddings, df_first_columns = compute_column_embeddings(
        df1, "the first database"
    )
    df_second_embeddings, df_second_columns = compute_column_embeddings(
        df2, "the second database"
    )

    # Get the similarity scores
    similarities = 1 - cdist(
        np.vstack(df_first_embeddings), np.vstack(df_second_embeddings), metric="cosine"
    )

    return similarities, df_first_columns, df_second_columns
