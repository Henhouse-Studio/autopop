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
    print("Reweighting columns...")
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


# Entry matching
def combine_dfs(
    df_base: pd.DataFrame,
    df_populate: pd.DataFrame,
    base_weights: dict,
    pop_weights: dict,
):
    """
    Combining the rows of two dictionaries based on the similarity scores.

    :param df_base: The dataframe to enrich (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment (pd.DataFrame).
    :param base_weights: The dictionary containing the masks for the base dataframe (dict):
    :param pop_weights: The dictionary containing the masks for the populator dataframe (dict):
    :param df_weights: The encoder type to use (defaults to 'all-MiniLM-L6-v2')
    :return: The merged dataframe (pd.Dataframe).
    """

    # Creating a mask based on the weight dict for the dataframes
    base_mask = list(base_weights.keys())
    pop_mask = list(pop_weights.keys())
    df_entries_base = df_base[base_mask]
    df_entries_pop = df_populate[pop_mask]

    scores = compute_similarity_entries_row(
        df_reweighting(df_entries_base, base_weights),
        df_reweighting(df_entries_pop, pop_weights),
    )

    return scores


def enrich_dataframes(
    df_ranked: dict,
    df_fact_ranked: dict,
    threshold: float = 0.7,
    model_encoder: str = "all-MiniLM-L6-v2",
):

    df_enriched = {}
    for key in df_ranked.keys():

        df_base = df_ranked[key][1]
        for key_fact in df_fact_ranked.keys():

            df_populate = df_fact_ranked[key_fact][1]

            # Check for matching column names between the base and populate dataframes
            matching_columns = df_base.columns.intersection(df_populate.columns)
            if matching_columns.empty:
                print(f"No matching columns between {key} and {key_fact}. Skipping...")
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

    print("Merging table pairs...")

    # Get the keys of the dataframes and the first dataframe
    table_names = list(df_ranked.keys())
    df_base = df_ranked[table_names[0]]
    base_weights = dict_weights[table_names[0]]

    # Iterate through the remaining dataframes and merge them
    for table_name in table_names[1:]:

        print(f"Merging '{table_name}' with the current base dataframe")
        df_populate = df_ranked[table_name]
        pop_weights = dict_weights[table_name]
        df_combined = combine_dfs(df_base, df_populate, base_weights, pop_weights)

        sys.exit()

        df_combined.to_csv("combined.csv")
        if df_combined.loc[:, "score"].mean() > args.threshold:
            df_base = df_combined

    # # Remove columns which are the same
    # final_df = remove_duplicates(df_base)

    # # Rename columns in case there are similar names
    # final_df = rename_columns(final_df, api_key=api_key)

    # # print(final_df)

    # return final_df
