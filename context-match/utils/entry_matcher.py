import sys
import pprint
import argparse
import pandas as pd
import linktransformer as lt
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


def group_scores_with_indices(scores_dict: dict, std_factor: int = 1):
    """
    Groups scores for each entry based on standard deviation while preserving indices.

    :param scores_dict: A dictionary with keys as tuples (row, column) and values as scores.
    :param std_factor: A multiplier for the standard deviation to determine group boundaries.
    :return: A dictionary with keys as row indices and values as lists of grouped tuples (index, score).
    """
    grouped_scores = {}

    # Collecting scores by row index along with their original indices
    scores_by_row = {}
    for (row, col), score in scores_dict.items():

        if row not in scores_by_row:
            scores_by_row[row] = []

        scores_by_row[row].append(((row, col), score))

    # Grouping scores by standard deviation within each row
    for row, score_tuples in scores_by_row.items():

        # Sort scores in descending order while keeping the index
        score_tuples.sort(key=lambda x: x[1], reverse=True)
        scores = [score for _, score in score_tuples]
        std_dev = np.std(scores)

        current_group = [score_tuples[0]]
        groups = [current_group]

        for i in range(1, len(score_tuples)):
            if (
                abs(score_tuples[i][1] - np.mean([x[1] for x in current_group]))
                <= std_dev * std_factor
            ):
                current_group.append(score_tuples[i])
            else:
                current_group = [score_tuples[i]]
                groups.append(current_group)

        grouped_scores[row] = groups

    pprint.pprint(grouped_scores)

    return grouped_scores


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


def merge_and_average_dicts(dict1, dict2):

    merged_dict = {}
    # Get all unique keys from both dictionaries
    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    for key in all_keys:

        if key in dict1 and key in dict2:
            # If the key is in both dictionaries, average the values
            merged_dict[key] = int((dict1[key] + dict2[key]) / 2)

        elif key in dict1:
            # If the key is only in dict1, take the value from dict1
            merged_dict[key] = dict1[key]

        elif key in dict2:
            # If the key is only in dict2, take the value from dict2
            merged_dict[key] = dict2[key]

    return merged_dict


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
    :return: The merged dataframe (pd.DataFrame) and the combined_weights (dict) between the two dataframes.

    * Note on pd.update:
        - If matched_base has a NaN value in the Location column, and matched_populate has a
          corresponding non-NaN value, matched_base will be updated with the value from matched_populate.
        - If matched_base already has a non-NaN value, it will not be overwritten because overwrite=False.

        # Update matched base DataFrame with populate DataFrame values
        # matched_base.update(matched_populate, overwrite=False)
    """

    # Creating a mask based on the weight dict for the dataframes
    df_entries_base = df_base[list(base_weights.keys())]
    df_entries_pop = df_populate[list(pop_weights.keys())]

    print("Computing between-row similarities...")
    scores = compute_similarity_entries_row(
        df_reweighting(df_entries_base, base_weights),
        df_reweighting(df_entries_pop, pop_weights),
    )

    # Filter the scores by group
    scores_f = filter_row_matches(scores)

    matched_base_indices = [i for i, _ in scores_f.keys()]
    matched_populate_indices = [j for _, j in scores_f.keys()]

    print("matched_base_indices", matched_base_indices )
    print("matched_populate_indices", matched_populate_indices )

    # Extract the corresponding matching rows from each DataFrame using the index pairs
    matched_base = df_base.loc[matched_base_indices].reset_index(drop=True)
    matched_populate = df_populate.loc[matched_populate_indices].reset_index(drop=True)

    # Combine matched DataFrames side by side and add confidence values
    matched_df = pd.concat(
        [
            matched_base,
            matched_populate.drop(columns=matched_base.columns, errors="ignore"),
        ],
        axis=1,
    )
    matched_df["conf_values"] = list(scores_f.values())

    # Filter by confidence threshold
    threshold = (1 - tolerance) * matched_df["conf_values"].quantile(0.25)
    matched_df = matched_df[matched_df["conf_values"] >= threshold]

    # Identify unmatched rows and assign NaN for missing columns
    unmatched_base = df_base.loc[~df_base.index.isin(matched_base_indices)]
    unmatched_populate = df_populate.loc[
        ~df_populate.index.isin(matched_populate_indices)
    ]

    # Append unmatched rows with NaN filled columns
    final_df = pd.concat(
        [
            matched_df,
            unmatched_base.assign(
                **{
                    col: None
                    for col in df_populate.columns
                    if col not in unmatched_base.columns
                }
            ),
            unmatched_populate.assign(
                **{
                    col: None
                    for col in df_base.columns
                    if col not in unmatched_populate.columns
                }
            ),
        ],
        ignore_index=True,
    )

    # Ensure all unmatched rows have a 'conf_values' column with 0 as a default value
    final_df["conf_values"].fillna(0, inplace=True)
    final_df.to_csv("merged.csv", index=False)

    sys.exit()

    # Combine the dictionary weights for merging later if needed
    combined_weights = merge_and_average_dicts(base_weights, pop_weights)

    return final_df, combined_weights


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
        df_combined, new_weights = combine_dfs(
            df_base, df_populate, base_weights, pop_weights, tolerance=args.tolerance
        )

        # pprint.pprint(new_weights)
        # df_combined.to_csv("combined.csv")

        if df_combined.loc[:, "conf_values"].mean() > args.matching_threshold:
            df_base = df_combined
            base_weights = new_weights

    # Rename columns in case there are similar names
    final_df = rename_columns(df_base, api_key=api_key)

    # print(final_df)

    return final_df
