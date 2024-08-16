import rapidfuzz
import pandas as pd


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

    # Create a new Series with adjusted confidence values
    adjusted_conf_values = df.apply(adjust_confidence, axis=1)

    # Use loc to update the 'conf_values' column
    df.loc[:, "conf_values"] = adjusted_conf_values

    return df


# Example usage:
if __name__ == "__main__":

    data = {
        "name": ["John Doe", "John Doe"],
        "name2": ["John D.", "Mason Valencia"],
        "conf_values": [0.75, 0.61],
    }
    df = pd.DataFrame(data)

    result = fuzzy_entry_rescorer(df)
    print(result)