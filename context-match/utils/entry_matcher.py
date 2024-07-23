import pandas as pd 

# Entry matching
def combine_dfs(
    df_base: pd.DataFrame, df_populate: pd.DataFrame, filtered_similarity_scores: dict
):
    print("Matching the entries...")
    merged_data = []

    # Iterate over the pointers to merge the rows
    pointers = list(filtered_similarity_scores.keys())
    for i, j in pointers:

        # Get the row from df_base
        df_base_row = df_base.iloc[i].copy()
        # Get the row from df_populate and rename its columns
        df_populate_row = df_populate.iloc[j].copy()
        df_populate_row.index = [f"{col}_df2" for col in df_populate_row.index]

        # Concatenate the rows along the columns
        merged_row = pd.concat([df_base_row, df_populate_row])
        merged_data.append(merged_row)

    # Convert the list of merged rows to a DataFrame
    result_df = pd.DataFrame(merged_data)
    print("Matching done!\n")

    return result_df