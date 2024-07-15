import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process


def get_similar_names(name, dataframe, column_name="name", top_n=5):
    """
    Get the most similar names from a dataframe.

    :param name: The name to compare.
    :param dataframe: The dataframe containing the names.
    :param column_name: The column name in the dataframe containing the names.
    :param top_n: The number of top similar names to return.
    :return: A list of tuples containing the top similar names and their similarity score.
    """

    names = dataframe[column_name].tolist()
    similar_names = process.extract(name, names, scorer=fuzz.ratio, limit=top_n)

    return similar_names


# Example usage:
if __name__ == "__main__":

    data = {
        "name": [
            "John Doe",
            "Johnny Depp",
            "Jonathan Davis",
            "Joan Baez",
            "Johanna Mason",
            "Johan Cruyff",
            "Jon Snow",
        ]
    }
    df = pd.DataFrame(data)

    input_name = "John D."
    result = get_similar_names(input_name, df)
    print(f"The 5 most similar names to '{input_name}' are:")
    for entry in result:
        print(f"{entry[0]}: {entry[1]}")
