import pandas as pd 

# Entry matching
def combine_dfs(
    df_base: pd.DataFrame, df_populate: pd.DataFrame, filtered_similarity_scores: dict
):
    print("Matching the entries...")
    merged_data = []

    # Iterate over the pointers to merge the rows
    pointers = list(filtered_similarity_scores.keys())

    # Get the first element of pointers into a list 
    pointers_base = [pointer[0] for pointer in pointers]
    remaining_pointers = [(index, "empty") for index in range(len(df_base)) if index not in pointers_base]
    pointers += remaining_pointers

    # sort the pointers by descending order by the first element of the tuple
    pointers = sorted(pointers, key=lambda x: x[0])
    
    # create an emtpy dataframe with col size as df_pop_col_size
    df_pop_col_size = len(df_populate.columns)
    empty_row = pd.Series([None] * df_pop_col_size, index=[f"{col}_df2" for col in df_populate.columns])    
    
    for i, j in pointers:

         # Get the row from df_base
        df_base_row = df_base.iloc[i].copy()

        if j != 'empty':
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