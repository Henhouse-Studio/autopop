import sys
import pprint
import argparse
import itertools
import pandas as pd
import linktransformer as lt
from typing import Tuple
from utils.prompt_to_openai import *
from utils.compute_similarity import *
from utils.fetch_table_notion import *


def df_reweighting(df: pd.DataFrame, weights: dict):
    """
    Multiply the dataframe columns based on the weights for encoding.

    :param df: The dataframe to reweight (pd.DataFrame).
    :param weights: The weights of each column (dict in the form of "colname: weight(int)").
    :return: The reweighted dataframe (pd.DataFrame).
    """
    # print("Reweighting columns...")
    # Create a copy of the dataframe to avoid modifying the original one
    df_reweighted = df.copy()

    for colname, weight in weights.items():
        # Skip columns with a weight of 0
        if weight == 0:
            continue

        for n_weight in range(weight):
            new_colname = f"{colname}_{n_weight}"
            # Use .loc to avoid the SettingWithCopyWarning
            df_reweighted.loc[:, new_colname] = df_reweighted[colname]

    return df_reweighted


def filter_row_matches(scores_dict: dict, std_factor: int = 1):
    """
    Get the top group of scores for each row based on standard deviation while preserving indices.

    :param scores_dict: A dictionary with keys as tuples (row, column) and values as scores.
    :param std_factor: A multiplier for the standard deviation to determine group boundaries.
    :return: A dictionary with keys as (row, column) tuples and values as scores, representing the top group.
    """
    top_group_scores = {}

    # Collecting scores by row index along with their original indices
    scores_by_row = {}
    for (row, col), score in scores_dict.items():

        if row not in scores_by_row:
            scores_by_row[row] = []

        scores_by_row[row].append(((row, col), round(score, 4)))

    # Selecting the top group based on standard deviation within each row
    for row, score_tuples in scores_by_row.items():
        # Sort scores in descending order while keeping the index
        score_tuples.sort(key=lambda x: x[1], reverse=True)
        scores = [score for _, score in score_tuples]
        std_dev = np.std(scores)

        top_group = [score_tuples[0]]

        for i in range(1, len(score_tuples)):
            if (
                abs(score_tuples[i][1] - np.mean([x[1] for x in top_group]))
                <= std_dev * std_factor
            ):
                top_group.append(score_tuples[i])
            else:
                break

        # Add top group to the final dictionary
        for index, score in top_group:
            top_group_scores[index] = score

    return top_group_scores


# Entry matching
def combine_dfs(
    df_base: pd.DataFrame,
    df_populate: pd.DataFrame,
    base_weights: dict,
    pop_weights: dict,
    tolerance: float = 0.15,
):
    """
    Combining the rows of two dataframes based on similarity scores using merge.

    :param df_base: The dataframe to enrich (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment (pd.DataFrame).
    :param base_weights: The dictionary containing the masks for the base dataframe (dict).
    :param pop_weights: The dictionary containing the masks for the populator dataframe (dict).
    :param df_weights: The encoder type to use (defaults to 'all-MiniLM-L6-v2').
    :param tolerance: How much to allow for potentially inaccurate matches (defaults to 0.15).
    The higher the tolerance the more indirect matches are allowed.
    :return: The merged dataframe (pd.DataFrame).
    """

    # Creating a mask based on the weight dict for the dataframes
    base_mask = list(base_weights.keys())
    pop_mask = list(pop_weights.keys())
    df_entries_base = df_base[base_mask]
    df_entries_pop = df_populate[pop_mask]

    print("Computing between-row similarities...")
    scores = compute_similarity_entries_row(
        df_reweighting(df_entries_base, base_weights),
        df_reweighting(df_entries_pop, pop_weights),
    )

    # Filter the scores by group
    scores_f = filter_row_matches(scores)

    # Extract the corresponding rows from each DataFrame using the index pairs in the dictionary
    rows_df1 = df_base.loc[[i for i, _ in scores_f.keys()]]
    rows_df2 = df_populate.loc[[j for _, j in scores_f.keys()]]
    conf_values = pd.DataFrame(list(scores_f.values()), columns=["conf_values"])

    # Reset index to prepare for merging
    rows_df1.reset_index(drop=True, inplace=True)
    rows_df2.reset_index(drop=True, inplace=True)
    conf_values.reset_index(drop=True, inplace=True)

    # Merge the dataframes on the reset index
    combined_df = pd.merge(
        rows_df1,
        rows_df2,
        left_index=True,
        right_index=True,
        suffixes=("_base", "_pop"),
    )

    # Add the confidence values to the merged dataframe
    combined_df["conf_values"] = conf_values

    # Minmax scaled threshold
    threshold = (1 - tolerance) * conf_values.quantile(0.25)

    # Filter based on the confidence threshold
    final_df = combined_df[combined_df["conf_values"] >= threshold.values[0]]
    final_df.to_csv("merged.csv", index=False)

    return final_df


def enrich_dataframes(
    df_ranked: dict,
    df_fact_ranked: dict,
    threshold: float = 0.7,
    model_encoder: str = "all-MiniLM-L6-v2",
):
    print("\nEnriching dataframes with Fact Tables...")

    df_enriched = {}
    for key in df_ranked.keys():

        df_base = df_ranked[key][1]
        for key_fact in df_fact_ranked.keys():

            df_populate = df_fact_ranked[key_fact][1]

            # Check for matching column names between the base and populate dataframes
            matching_columns = df_base.columns.intersection(df_populate.columns)
            if matching_columns.empty:
                print(
                    f"No matching columns between '{key}' and '{key_fact}'. Skipping..."
                )
                continue

            # Combine the dataframes if there are matching columns
            df_combined = lt.merge(df_base, df_populate, model=model_encoder)
            df_combined = df_combined.drop(["id_lt_x", "id_lt_y"], axis=1)

            if df_combined.loc[:, "score"].mean() > threshold:
                df_base = df_combined

        df_enriched[key] = df_base

    return df_enriched


def merge_top_k(
    df_ranked: dict, dict_weights: dict, api_key: str, args: argparse.Namespace
):

    print("\nMerging table pairs...")

    # Get the keys of the dataframes and the first dataframe
    table_names = list(df_ranked.keys())
    df_base = df_ranked[table_names[0]]
    base_weights = dict_weights[table_names[0]]

    # Iterate through the remaining dataframes and merge them
    for table_name in table_names[1:]:

        print(f"Merging current base dataframe with '{table_name}'")
        df_populate = df_ranked[table_name]
        pop_weights = dict_weights[table_name]
        df_combined = combine_dfs(df_base, df_populate, base_weights, pop_weights)

        # print(df_combined)
        # df_combined.to_csv("combined.csv")

        if df_combined.loc[:, "conf_values"].mean() > args.threshold:
            df_base = df_combined

    # Remove columns which are the same
    final_df = remove_duplicates(df_base)

    # Rename columns in case there are similar names
    final_df = rename_columns(final_df, api_key=api_key)

    # print(final_df)

    return final_df
