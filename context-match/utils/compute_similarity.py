import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import util
from scipy.spatial.distance import cdist
from utils.make_embeddings import compute_embedding
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


def compute_similarity_entries_row(
    df_base: pd.DataFrame, df_populate: pd.DataFrame, verbose: bool = False
):
    """
    Compute similarity scores between the rows of two dataframes.

    This function calculates the similarity between each row in `df_base` and each row in `df_populate`
    by generating embeddings for the rows and then computing similarity scores between these embeddings.
    The result is a dictionary where each key is a tuple representing a pair of row indices (from `df_base` and
    `df_populate`), and the corresponding value is the similarity score between those rows.

    The similarity scores are sorted primarily by the index of `df_base` and secondarily by the score in
    descending order.

    :param df_base: The dataframe to be enriched, containing the rows for which similarity needs to be computed (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment, containing the rows to compare against `df_base` (pd.DataFrame).
    :param verbose: Whether to print the computed similarity scores (bool, default = False).

    :return: A dictionary containing the similarity scores, with keys as tuples `(idx_df1, idx_df2)`
             and values as similarity scores (float). The keys represent the row indices in `df_base`
             and `df_populate`, respectively (dict).
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
