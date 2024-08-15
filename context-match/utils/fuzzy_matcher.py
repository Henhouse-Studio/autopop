import rapidfuzz
import pandas as pd

# def fuzzy_entry_rescorer(
#     df: pd.DataFrame, threshold: float = 0.5, penalty_factor: float = 0.35
# ) -> pd.DataFrame:
#     """
#     Compare the entries between columns to see if they match and penalize
#     the confidence scores accordingly if they don't.

#     :param df: The dataframe containing the columns to match between.
#     :param threshold: The threshold to detect the non-matches.
#     :param penalty_factor: The factor by which to penalize the confidence score for mismatches.
#     :return: The rescored dataframe.
#     """

#     colnames = list(df.columns)

#     # Calculate the matching ratio
#     df = df.copy()  # Make an explicit copy to avoid SettingWithCopyWarning
#     df["matching_ratio"] = df.apply(
#         lambda x: rapidfuzz.fuzz.QRatio(x[colnames[1]], x[colnames[3]]), axis=1
#     )

#     # Apply a penalty to the confidence values where the matching ratio is below the threshold
#     df.loc[df["matching_ratio"] < (threshold * 100), "conf_values"] -= df.loc[
#         df["matching_ratio"] < (threshold * 100), "matching_ratio"
#     ].apply(lambda ratio: penalty_factor * (1 - ratio / 100))

#     # Ensure that confidence values don't fall below 0
#     df["conf_values"] = df["conf_values"].clip(lower=0)

#     # Remove the matching_ratio column
#     df.drop(columns=["matching_ratio"], inplace=True)

#     return df

# # Example usage:
# if __name__ == "__main__":

#     data = {
#         "name": ["John Doe", "John Doe"],
#         "name2": ["John D.", "Mason Valencia"],
#         "conf_values": [0.75, 0.61],
#     }
#     df = pd.DataFrame(data)

#     result = fuzzy_entry_rescorer(df)
#     print(result)


def fuzzy_entry_rescorer(
    df: pd.DataFrame, threshold: float = 0.5, penalty_factor: float = 0.35
):
    """
    Compare the entries between columns to see if they match and penalize
    the confidence scores accordingly if they don't.
    If one of the strings is empty, set the confidence score to 0.

    :param df: The dataframe containing the columns to match between.
    :param threshold: The threshold to detect the non-matches.
    :param penalty_factor: The factor by which to penalize the confidence score for mismatches.
    :return: The rescored dataframe.
    """

    colnames = list(df.columns)

    # Calculate the matching ratio and adjust confidence values
    def adjust_confidence(row):
        str1, str2 = row[colnames[1]], row[colnames[3]]
        
        # Check if one of the strings is empty
        if not str1 or not str2:
            return 0
        
        # Calculate matching ratio
        matching_ratio = rapidfuzz.fuzz.QRatio(str1, str2)
        
        # Penalize confidence value if matching ratio is below the threshold
        if matching_ratio < threshold * 100:
            penalty = penalty_factor * (1 - matching_ratio / 100)
            return max(row["conf_values"] - penalty, 0)
        
        return row["conf_values"]

    # Apply the adjust_confidence function to each row in the dataframe
    df["conf_values"] = df.apply(adjust_confidence, axis=1)

    return df