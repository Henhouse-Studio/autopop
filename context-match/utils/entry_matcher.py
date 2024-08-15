import sys
import pprint
import argparse
import pandas as pd
import linktransformer as lt
from utils.fuzzy_matcher import *
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
        if weight <= 0:
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
    """
    Merge two dictionaries by averaging the values for common keys.

    For each key in either dictionary:
    - If the key is present in both dict1 and dict2, its value in the resulting dictionary
      will be the average of the values from dict1 and dict2 (rounded down to an integer).
    - If the key is present only in dict1, its value will be taken from dict1.
    - If the key is present only in dict2, its value will be taken from dict2.

    :param dict1: The first dictionary, with keys as tuples and values as integers.
    :param dict2: The second dictionary, with keys as tuples and values as integers.
    :return: A new dictionary with keys as tuples and averaged or merged values.
    """

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


def generate_fuzzy_match_description(df_base: pd.DataFrame, df_populate: pd.DataFrame, 
                                     n_samples: int = 2, random_state: int = 42, verbose: bool = False
) -> str:
    """
    Generate a descriptive extract of two tables for fuzzy matching.

    Args:
        df_base (pd.DataFrame): The base DataFrame to sample from.
        df_populate (pd.DataFrame): The populate DataFrame to sample from.
        base_sample_size (int): Number of samples to take from the base DataFrame. Default is 2.
        populate_sample_size (int): Number of samples to take from the populate DataFrame. Default is 2.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        str: A formatted string representing a sample extract from both tables.
    """

    # Sample and extract from df_base
    col_names_base = list(df_base.columns)
    sample_base = df_base.sample(n=n_samples, random_state=random_state)
    description = "This is a extract from the Table 1:\n"

    for sample_value in range(len(sample_base)):
        description += f"[{sample_value}] Entry\n"
        for col_name in col_names_base:
            description += f"[Column Name]: {col_name}, [Value]: {sample_base[col_name].values[sample_value]}\n"

    # Sample and extract from df_populate
    col_names_populate = list(df_populate.columns)
    sample_populate = df_populate.sample(n=n_samples, random_state=random_state)
    description += "This is a extract from the Table 2:\n"

    for sample_value in range(len(sample_populate)):
        description += f"[{sample_value}] Entry\n"
        for col_name in col_names_populate:
            description += f"[Column Name]: {col_name}, [Value]: {sample_populate[col_name].values[sample_value]}\n"
    if verbose:
        print(description)
    return description


# Entry matching
def combine_dfs(
    prompt: str,
    df_base: pd.DataFrame,
    df_populate: pd.DataFrame,
    base_weights: dict,
    pop_weights: dict,
    tolerance: float = 0.05,
    api_key: str = None,
):
    """
    Combining the rows of two dataframes based on similarity scores using merge.

    :param prompt: The prompt to use for the OpenAI API (str).
    :param df_base: The dataframe to enrich (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment (pd.DataFrame).
    :param base_weights: The dictionary containing the masks for the base dataframe (dict).
    :param pop_weights: The dictionary containing the masks for the populator dataframe (dict).
    :param df_weights: The encoder type to use (defaults to 'all-MiniLM-L6-v2').
    :param tolerance: How much to allow for potentially inaccurate matches (defaults to 0.15).
                      The higher the tolerance the more indirect matches are allowed.
    :param api_key: The API key for renaming columns in the final dataframe (str).

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
    # (row, col) : score
    scores_f = filter_row_matches(scores) 
    # pprint.pprint(scores_f)

    matched_base_indices = [int(i) for i, _ in scores_f.keys()]
    matched_populate_indices = [int(j) for _, j in scores_f.keys()]

    # Extract the corresponding matching rows from each DataFrame using the index pairs
    matched_base = df_base.loc[matched_base_indices].reset_index(drop=True)
    matched_populate = df_populate.loc[matched_populate_indices].reset_index(drop=True)

    # add index column at beginning to keep track of the original index
    matched_base.insert(0, "index_base", matched_base_indices)
    matched_populate.insert(0, "index_pop", matched_populate_indices)

    # Combine matched DataFrames side by side and add confidence values
    matched_df = pd.concat(
        [
            matched_base,
            matched_populate.drop(columns=matched_base.columns, errors="ignore"),
        ],
        axis=1,
    )

    matched_df["conf_values"] = list(scores_f.values())
    # matched_df.to_csv("test.csv", index=False)

    # Filter by confidence threshold
    threshold = (1 - tolerance) * 0.48
    matched_df = matched_df[matched_df["conf_values"] >= threshold]

    # Filter the scores based on the threshold
    # (row, col) : scored_tresholded
    filtered_scores_f = {k: v for k, v in scores_f.items() if v >= threshold}

    matched_base_indices = [i for i, _ in filtered_scores_f.keys()] 
    matched_populate_indices = [j for _, j in filtered_scores_f.keys()] 

    # Identify unmatched rows and assign NaN for missing columns
    unmatched_base = df_base.loc[~df_base.index.isin(matched_base_indices)]
    unmatched_populate = df_populate.loc[
        ~df_populate.index.isin(matched_populate_indices)
    ]

    # add index column at beginning to keep track of the original index
    unmatched_base_indices = df_base.index.difference(matched_base_indices)
    unmatched_populate_indices = df_populate.index.difference(matched_populate_indices)

    unmatched_base.insert(0, "index_base", unmatched_base_indices)
    unmatched_populate.insert(0, "index_pop", unmatched_populate_indices)

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

    # Get matching column entities
    desc_of_tables = generate_fuzzy_match_description(df_base, df_populate, n_samples=2, verbose=False)
    matched_col_entities = get_column_names(prompt, desc_of_tables, api_key, verbose=True)

    # Select matched entities columns and drop rows that have NA values
    matched_entities_df = final_df[
        ["index_base"]
        + [matched_col_entities[0]]
        + ["index_pop"]
        + [matched_col_entities[1]]
        + ["conf_values"]
    ].dropna()
    repeated_entities_df = matched_entities_df[
        matched_entities_df.duplicated(subset=[matched_col_entities[0]], keep=False)
    ]
    repeated_entities_df.to_csv("repeated_entities.csv", index=False)

    rescored_repeated_entities_df = fuzzy_entry_rescorer(repeated_entities_df)
    rescored_repeated_entities_df.to_csv("rescored_repeated_entities.csv", index=False)
    final_df.update(rescored_repeated_entities_df, overwrite=True)
    final_df = final_df[final_df["conf_values"] >= threshold]
    final_df.to_csv("final_df.csv", index=False)


    # Find the missing indices in final_df from df_base and df_populate
    missing_base_indices = df_base.index.difference(final_df['index_base'].dropna().astype(int))
    missing_pop_indices = df_populate.index.difference(final_df['index_pop'].dropna().astype(int))

    # Extract the corresponding rows from df_base and df_populate
    missing_base_rows = df_base.loc[missing_base_indices].copy()
    missing_pop_rows = df_populate.loc[missing_pop_indices].copy()

    # Handle missing columns by adding NaN-filled columns
    for col in final_df.columns:
        if col not in missing_base_rows.columns:
            missing_base_rows[col] = None
        if col not in missing_pop_rows.columns:
            missing_pop_rows[col] = None

    # Concatenate missing rows with the final_df
    final_combined_df = pd.concat([final_df, missing_base_rows, missing_pop_rows], ignore_index=True)

    # Drop index_base and index_pop columns
    final_combined_df.drop(columns=['index_base', 'index_pop'], inplace=True)
    print(final_combined_df.index)

    # Save the final dataframe to a new CSV
    final_combined_df.to_csv('final_combined_df.csv', index=False)
    print("Merged dataframes!")

    # Combine the dictionary weights for merging later if needed
    combined_weights = merge_and_average_dicts(base_weights, pop_weights)

    return final_combined_df, combined_weights


def enrich_dataframes(
    df_ranked: dict,
    df_fact_ranked: dict,
    threshold: float = 0.7,
    model_encoder: str = "all-MiniLM-L6-v2",
):
    """
    Enrich dataframes by merging them with fact tables based on matching column names.

    This function iterates over ranked dataframes and fact tables, merges them where columns match,
    and retains the merged dataframe only if the average "score" exceeds the specified threshold.

    :param df_ranked: A dictionary of ranked dataframes, where keys are identifiers and values are tuples.
                      The second item in the tuple is the base dataframe to be enriched.
    :param df_fact_ranked: A dictionary of fact table dataframes, where keys are identifiers and values are tuples.
                           The second item in the tuple is the fact table dataframe used for enrichment.
    :param threshold: A float representing the minimum average "score" for the merged dataframe to be accepted.
                      If the average score is above the threshold, the dataframe is retained. Default is 0.7.
    :param model_encoder: A string representing the model used for merging dataframes. Default is "all-MiniLM-L6-v2".
    :return: A dictionary of enriched dataframes with the same keys as the original df_ranked dictionary.
             The base dataframe is updated with fact table data if column names match and the score threshold is met.
    """

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
    prompt:str, df_ranked: dict, dict_weights: dict, api_key: str, args: argparse.Namespace
):
    """
    Merge top-ranked dataframes sequentially based on matching criteria and weighted values.

    :param df_ranked: A dictionary where keys are table names and values are the ranked dataframes to be merged.
    :param dict_weights: A dictionary where keys are table names and values are the weights associated with each dataframe.
    :param api_key: A string representing the API key for renaming columns in the final dataframe.
    :param args: A Namespace object containing additional arguments like tolerance for merging and matching_threshold for
                 determining whether dataframes can be combined based on their confidence values.
    :return: The final merged dataframe with renamed columns.
    """

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
            prompt, df_base, df_populate, base_weights, pop_weights, 
            tolerance=args.tolerance, api_key=api_key
        )

        # Ensure that the rows actually do match, otherwise the dataframes are likely mismatched
        if (
            df_combined[df_combined["conf_values"] != 0]["conf_values"].mean()
            > args.matching_threshold
        ):
            df_base = df_combined
            base_weights = new_weights

    # Rename columns in case there are similar names
    final_df = rename_columns(df_base, api_key=api_key)

    # print(final_df)

    return final_df
