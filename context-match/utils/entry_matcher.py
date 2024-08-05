import sys
import pprint
import argparse
import itertools
import pandas as pd
from utils.compute_similarity import *
from utils.prompt_to_openai import *
from utils.fetch_table_notion import *


# Entry matching
def combine_dfs(
    df_base: pd.DataFrame, df_populate: pd.DataFrame, filtered_similarity_scores: dict
):
    """
    Combining the rows of two dictionaries based on the similarity scores.

    :param df_base: The dataframe to enrich (pd.DataFrame).
    :param df_populate: The dataframe used for enrichment (pd.DataFrame).
    :param filtered_similarity_scores: The similarity scores to match with (dict). Should be
    in the form of '(idx_df1, idx_df2): score' per entry.
    :return: The merged dataframe (pd.Dataframe).
    """

    print("Matching the entries...")
    merged_data = []

    # Iterate over the pointers to merge the rows
    pointers = list(filtered_similarity_scores.keys())

    # Get the first element of pointers into a list
    pointers_base = [pointer[0] for pointer in pointers]
    remaining_pointers = [
        (index, "empty") for index in range(len(df_base)) if index not in pointers_base
    ]
    pointers += remaining_pointers

    # Sort the pointers by descending order by the first element of the tuple
    pointers = sorted(pointers, key=lambda x: x[0])

    # Create an emtpy dataframe with col size as df_pop_col_size
    df_pop_col_size = len(df_populate.columns)
    empty_row = pd.Series(
        [None] * df_pop_col_size, index=[f"{col}_df2" for col in df_populate.columns]
    )

    for i, j in pointers:

        # Get the row from df_base
        df_base_row = df_base.iloc[i].copy()

        if j != "empty":
            # Get the row from df_populate and rename its columns
            df_populate_row = df_populate.iloc[j].copy()
            df_populate_row.index = [f"{col}_df2" for col in df_populate_row.index]
            merged_row = pd.concat([df_base_row, df_populate_row])

        else:
            merged_row = pd.concat([df_base_row, empty_row])

        # Concatenate the rows along the columns
        merged_data.append(merged_row)

    # Convert the list of merged rows to a DataFrame
    result_df = pd.DataFrame(merged_data)
    print("Matching done!\n")

    return result_df


def merge_top_k(df_ranked: dict, args: argparse.Namespace):

    print("\nMerging pairs of tables...")
    # Generate pairs
    keys = list(df_ranked.keys())
    table_pairs = list(itertools.combinations(keys, 2))

    for pair in table_pairs:

        print(pair)
        df_first = df_ranked[pair[0]][1]
        df_second = df_ranked[pair[1]][1]

        score_dict, _ = compute_similarity_entries_row(
            df_base=df_first, df_populate=df_second
        )

        # score_dict_col, _ = compute_similarity_entries_col(
        #     df_base=df_first, df_populate=df_second
        # )

    # # Threshold based on table size
    # threshold = 2 * args.threshold / len(df_second)

    # # Filter similarity scores based on threshold
    # filtered_similarity_scores = {k: v for k, v in score_dict.items() if v >= threshold}
    # sorted_similarity_scores = dict(sorted(score_dict.items(), ke=lambda x: x[1]))

    # for k, v in sorted(score_dict.items(), key=lambda item: (item[0][0], -item[1])):
    #     sorted_dict[k] = v

    pprint.pprint(score_dict)

    sys.exit()

    print(f"Found {len(filtered_similarity_scores)} matches!\n")

    final_df = combine_dfs(df_first, df_second, filtered_similarity_scores)
    # final_df = final_df.drop(f"{highest_similar_col_name}_df2", axis="columns")

    # Remove columns which are the same
    final_df = remove_duplicates(final_df)

    # Rename columns in case there are similar names
    final_df = rename_columns(final_df, api_key=OPENAI_TOKEN)

    final_df.to_csv("out.csv", index=False)
    # print(final_df)

    return final_df
