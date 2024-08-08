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


# Entry matching
def combine_dfs(
    df_base: pd.DataFrame,
    df_populate: pd.DataFrame,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    model_encoder: str = "all-MiniLM-L6-v2",
):
    """
    Combining the rows of two dictionaries based on the similarity scores.

    :param df_base: The dataframe to enrich (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment (pd.DataFrame).
    :param model: The encoder type to use (defaults to 'all-MiniLM-L6-v2')
    :return: The merged dataframe (pd.Dataframe).
    """

    merged_data = lt.merge(df_base, df_populate, model=model_encoder, suffixes=suffixes)

    return merged_data


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
            df_combined = combine_dfs(df_base, df_populate, model_encoder=model_encoder)
            df_combined = df_combined.drop(["id_lt_x", "id_lt_y"], axis=1)

            if df_combined.loc[:, "score"].mean() > threshold:
                df_base = df_combined

        df_enriched[key] = df_base

    return df_enriched


# TODO: Cleanup colnames
def merge_top_k(df_ranked: dict, api_key: str, args: argparse.Namespace):

    print("Merging table pairs...")

    # Get the keys of the dataframes and the first dataframe
    keys = list(df_ranked.keys())
    df_base = df_ranked[keys[0]]

    # Iterate through the remaining dataframes and merge them
    for key in keys[1:]:

        print(f"Merging '{key}' with the current base dataframe")
        df_populate = df_ranked[key]
        df_combined = combine_dfs(
            df_base, df_populate, model_encoder=args.model_encoder
        )

        df_combined.to_csv("combined.csv")
        if df_combined.loc[:, "score"].mean() > args.threshold:
            df_base = df_combined

    # Remove columns which are the same
    final_df = remove_duplicates(df_base)

    # Rename columns in case there are similar names
    final_df = rename_columns(final_df, api_key=api_key)

    # print(final_df)

    return final_df
