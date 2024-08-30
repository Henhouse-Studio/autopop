import os
import csv
import uuid
import numpy as np
import pandas as pd
from utils.constants import *


def save_dataframe(df: pd.DataFrame, df_collection: dict):
    """
    Save a DataFrame and record its reference in a CSV file.

    This function saves the given DataFrame to a CSV file with a unique name
    and logs this reference along with the names of the contributing DataFrames
    in a separate CSV file.

    :param df: The DataFrame to be saved (pd.DataFrame).
    :param df_collection: A dictionary containing the DataFrames involved in the merge
                          where keys are names of the DataFrames (dict).
    """

    # Create the file name
    merge_name = f"{uuid.uuid4().hex}.csv"

    # Define the content of the new row
    base_names = list(df_collection.keys())
    # base_names.sort()
    content = [
        "",
        "",
        "",
        "",
        "",
        merge_name,
    ]
    content[: len(base_names)] = base_names

    # Save the reference to the file
    with open(MERGED_DIR, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(content)

    # Create the .csv for the merge
    savedir = os.path.join(SAVE_MERGE_DIR, merge_name)
    df.to_csv(savedir, index=False)

    print(f"Saved dataframe combination to {merge_name}!")


def check_merged_table(df_collection: dict):
    """
    Check if a merged DataFrame already exists based on the collection of DataFrame names.

    This function reads a record of previously saved DataFrames and checks if a
    merge with the given DataFrame names already exists. If it does, it loads and
    returns the existing merged DataFrame.

    :param df_collection: A dictionary containing the DataFrames involved in the merge
                          where keys are names of the DataFrames (dict).
    :return: The existing merged DataFrame if a match is found, otherwise None (pd.DataFrame or None).
    """

    # Get the keys as a matching list and sort them
    entry = list(df_collection.keys())
    # entry.sort()

    # Ensure the rest of the columns are registered as NaN
    entry_full = [np.nan, np.nan, np.nan, np.nan, np.nan]
    entry_full[: len(entry)] = entry

    # Check if the cache merge dir exists
    if not os.path.isfile(MERGED_DIR):
        with open(MERGED_DIR, "w", newline="") as f:
            writer = csv.writer(f)
            field = ["table_1", "table_2", "table_3", "table_4", "table_5", "df_name"]
            writer.writerow(field)

    df = pd.read_csv(MERGED_DIR)

    # Select only the first five columns (table_1 to table_5)
    df_new = df[["table_1", "table_2", "table_3", "table_4", "table_5"]]

    # Convert empty strings to NaN
    df_new = df_new.replace("", np.nan)

    # Convert each row to a list and compare with entry_full
    def row_matches(row):
        return all(
            (x == y or (pd.isna(x) and pd.isna(y))) for x, y in zip(row, entry_full)
        )

    matches = df_new.apply(row_matches, axis=1)

    if matches.any():
        print("Existing entry found!")
        matching_index = matches.idxmax()  # Get the index of the first match
        merge_cache = df.at[matching_index, "df_name"]
        load_dir = os.path.join(SAVE_MERGE_DIR, merge_cache)
        merge_df = pd.read_csv(load_dir)

        return merge_df

    else:
        return None
